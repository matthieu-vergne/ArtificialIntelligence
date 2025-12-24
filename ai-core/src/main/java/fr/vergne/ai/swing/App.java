package fr.vergne.ai.swing;

import static fr.vergne.ai.utils.LambdaUtils.memoize;
import static java.util.stream.Collectors.toMap;

import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.Paint;
import java.awt.event.ActionEvent;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.JTextField;
import javax.swing.JToggleButton;
import javax.swing.SwingUtilities;
import javax.swing.filechooser.FileNameExtensionFilter;

import org.apache.batik.anim.dom.SAXSVGDocumentFactory;
import org.apache.batik.swing.JSVGCanvas;
import org.apache.batik.util.XMLResourceDescriptor;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.LogarithmicAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.LookupPaintScale;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.ui.RectangleAnchor;
import org.jfree.chart.ui.RectangleEdge;
import org.jfree.data.DomainOrder;
import org.jfree.data.general.DatasetChangeListener;
import org.jfree.data.general.DatasetGroup;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.data.xy.XYZDataset;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.svg.SVGDocument;
import org.w3c.dom.svg.SVGSVGElement;

import fr.vergne.ai.neuralnet.NeuralNet.Input;
import fr.vergne.ai.neuralnet.NeuralNet.Layer;
import fr.vergne.ai.neuralnet.NeuralNet.MLP;
import fr.vergne.ai.neuralnet.NeuralNet.NeuralNetBrowser;
import fr.vergne.ai.neuralnet.NeuralNet.Neuron;
import fr.vergne.ai.neuralnet.NeuralNet.Operator;
import fr.vergne.ai.neuralnet.NeuralNet.Value;
import fr.vergne.ai.utils.LambdaUtils;

@SuppressWarnings("serial")
public class App extends JFrame {
	public App(Conf conf, Map<List<Double>, Double> dataset, MLP mlp, Supplier<RoundData> mlpRound) {
		this.setTitle("MLP");

		// TODO Create conf panel
		// TODO Create dataset panel
		Parts mlpParts = createMlpPanel(conf.neuralNetConf(), mlp);
		Parts visualParts = createVisual(conf.visualConf(), dataset);
		Parts lossPlotParts = createLossPlot(conf.lossPlotConf());
		Parts timePlotParts = createTimePlot(conf.timePlotConf());

		JTabbedPane tabs = new JTabbedPane();
		tabs.add(mlpParts.panel(), "MLP");
		tabs.add(visualParts.panel(), "Visual");
		tabs.add(lossPlotParts.panel(), "Loss plot");
		tabs.add(timePlotParts.panel(), "Time plot");

		Consumer<List<RoundResult>> roundConsumer = mlpParts.panelUpdater()//
				.andThen(visualParts.panelUpdater())//
				.andThen(lossPlotParts.panelUpdater())//
				.andThen(timePlotParts.panelUpdater())//
		;

		JPanel trainPanel = createTrainPanel(conf.trainConf(), mlpRound, roundConsumer);

		this.setLayout(new GridBagLayout());
		GridBagConstraints constraints = new GridBagConstraints();
		constraints.gridx = 0;
		constraints.gridy = GridBagConstraints.RELATIVE;
		constraints.weightx = 1.0;
		constraints.weighty = 1.0;
		constraints.fill = GridBagConstraints.BOTH;
		this.add(tabs, constraints);
		constraints.weightx = 1.0;
		constraints.weighty = 0.0;
		constraints.fill = GridBagConstraints.HORIZONTAL;
		this.add(trainPanel, constraints);
	}

	private Parts createMlpPanel(NeuralNetConf neuralNetConf, MLP mlp) {
		// TODO Factor DOT producer + NeuralNetBrowser + SVG updater (node IDs, etc.)
		// TODO Support NeuralNet building
		Function<Input, String> inputIdSupplier = memoize(prefixedCounterId("I"));
		Function<Neuron, String> neuronIdSupplier = memoize(prefixedCounterId("N"));
		Function<Layer, String> layerIdSupplier = memoize(prefixedCounterId("L"));

		String fileName = "graph";
		Path dotPath = createTempPath(fileName, "dot");
		createMlpDot(neuralNetConf, mlp, dotPath, inputIdSupplier, neuronIdSupplier, layerIdSupplier);

		Path svgPath = createTempPath(fileName, "svg");
		dotToFile(dotPath, svgPath, "svg");

		String saxParser = XMLResourceDescriptor.getXMLParserClassName();
		SAXSVGDocumentFactory factory = new SAXSVGDocumentFactory(saxParser);
		SVGDocument createdDocument;
		try {
			createdDocument = factory.createSVGDocument(svgPath.toString());
		} catch (IOException cause) {
			throw new RuntimeException(cause);
		}

		JSVGCanvas svgCanvas = new JSVGCanvas();
		svgCanvas.setDocumentState(JSVGCanvas.ALWAYS_DYNAMIC);
		svgCanvas.setSVGDocument(createdDocument);
		// A deep copy might be done, so retrieve actually stored instance
		SVGDocument svgDocument = svgCanvas.getSVGDocument();
		// We will work on it from its root
		SVGSVGElement root = svgDocument.getRootElement();

		Consumer<List<RoundResult>> mlpUpdater = _ -> {
			/*
			 * As per Batik documentation, for a dynamic SVG to properly trigger its
			 * repainting, the changes must be performed in the updater manager queue.
			 */
			svgCanvas.getUpdateManager().getUpdateRunnableQueue().invokeLater(() -> {
				int decimals = neuralNetConf.displayedDecimals();
				Function<Double, String> parameterFormatter = value -> {
					// TODO Use app locale?
					return String.format(Locale.US, "%." + decimals + "f", value);
				};
				NeuralNetBrowser mlpBrowser = NeuralNetBrowser.forMlp(mlp);
				streamOf(root.getElementsByTagName("g")).forEach(groupNode -> {
					String title = titleOf(groupNode);
					if (title.matches("[IN][0-9]+->N[0-9]+")) {
						int separatorStart = title.indexOf("->");
						int separatorEnd = separatorStart + 2;
						String inputId = title.substring(0, separatorStart);
						String outputId = title.substring(separatorEnd);
						double weight = mlpBrowser.neuron(outputId).weightWith(inputId);
						Node weightNode = findFirstChild(groupNode, "text").orElseThrow();
						weightNode.setTextContent(parameterFormatter.apply(weight));
					} else if (title.matches("N[0-9]+")) {
						String neuronId = title;
						double bias = mlpBrowser.neuron(neuronId).bias();
						// First text is node name, bias is next one
						Node biasNode = findSecondChild(groupNode, "text").orElseThrow();
						biasNode.setTextContent(parameterFormatter.apply(bias));
					} else {
						// Ignore
					}
				});
			});
			;
		};

		JPanel exportPanel = createExportPanel(neuralNetConf, mlp, inputIdSupplier, neuronIdSupplier, layerIdSupplier);

		JPanel mlpPanel = new JPanel();
		mlpPanel.setLayout(new GridBagLayout());
		GridBagConstraints constraints = new GridBagConstraints();
		constraints.gridx = 1;
		constraints.gridy = GridBagConstraints.RELATIVE;
		constraints.weightx = 1.0;
		constraints.weighty = 1.0;
		constraints.fill = GridBagConstraints.BOTH;
		mlpPanel.add(svgCanvas, constraints);
		constraints.weighty = 0.0;
		mlpPanel.add(exportPanel, constraints);

		return new Parts(mlpPanel, mlpUpdater);
	}

	private JPanel createExportPanel(NeuralNetConf neuralNetConf, MLP mlp, Function<Input, String> inputIdSupplier,
			Function<Neuron, String> neuronIdSupplier, Function<Layer, String> layerIdSupplier) {
		JPanel exportPanel = new JPanel();
		exportPanel.setLayout(new FlowLayout(FlowLayout.CENTER));
		exportPanel.add(new JButton(new AbstractAction("Export") {

			@Override
			public void actionPerformed(ActionEvent e) {
				JFileChooser fileChooser = new JFileChooser();
				fileChooser.setDialogTitle("Export Data");
				fileChooser.setCurrentDirectory(new File("."));
				fileChooser.addChoosableFileFilter(new FileNameExtensionFilter("DOT", "dot"));
				fileChooser.addChoosableFileFilter(new FileNameExtensionFilter("SVG", "svg"));
				fileChooser.addChoosableFileFilter(new FileNameExtensionFilter("PNG", "png"));
				fileChooser.addChoosableFileFilter(new FileNameExtensionFilter("PDF", "pdf"));
				fileChooser.addPropertyChangeListener("fileFilterChanged", evt -> {
					Path directory = fileChooser.getCurrentDirectory().toPath();
					FileNameExtensionFilter newFilter = (FileNameExtensionFilter) evt.getNewValue();
					String newExt = newFilter.getExtensions()[0];
					String newFileName = "neuralnet." + newExt;
					Path newPath = directory.resolve(newFileName);
					fileChooser.setSelectedFile(newPath.toFile());
				});
				// Remove default file filter because we focus on supported formats
				// Having it after setting the listener triggers it to auto-name the file
				fileChooser.removeChoosableFileFilter(fileChooser.getAcceptAllFileFilter());

				int result = fileChooser.showSaveDialog(exportPanel);
				if (result == JFileChooser.APPROVE_OPTION) {
					File file = fileChooser.getSelectedFile();
					String fileName = file.getName();
					if (fileName.endsWith(".dot")) {
						if (existsAndOverrideRejected(file)) {
							return; // User chose not to overwrite, do nothing
						}
						createMlpDot(neuralNetConf, mlp, file.toPath(), inputIdSupplier, neuronIdSupplier,
								layerIdSupplier);
					} else if (fileName.endsWith(".svg") || fileName.endsWith(".pdf") || fileName.endsWith(".png")) {
						if (existsAndOverrideRejected(file)) {
							return; // User chose not to overwrite, do nothing
						}
						Path tempDotPath = createTempPath(fileName, "dot");
						createMlpDot(neuralNetConf, mlp, tempDotPath, inputIdSupplier, neuronIdSupplier,
								layerIdSupplier);
						String ext = fileName.substring(fileName.length() - 3);
						dotToFile(tempDotPath, file.toPath(), ext);
					} else {
						throw new IllegalArgumentException("Unsupported format: " + fileName);
					}
				}
			}

			private boolean existsAndOverrideRejected(File file) {
				if (!file.exists()) {
					return false;
				}

				int overrideConfirmation = JOptionPane.showConfirmDialog(exportPanel,
						"The file already exists. Do you want to overwrite it?", "Overwrite File",
						JOptionPane.YES_NO_OPTION);
				if (overrideConfirmation == JOptionPane.YES_OPTION) {
					return false;
				}

				return true;
			}
		}));
		return exportPanel;
	}

	private static Parts createVisual(VisualConf visualConf, Map<List<Double>, Double> dataset) {
		BiFunction<VisualDatasetConf, Integer, List<Double>> valuesExtractor = (datasetConf1, index) -> {
			return dataset.entrySet().stream()//
					.filter(entry -> datasetConf1.predicate().test(entry.getValue()))//
					.map(Entry::getKey)//
					.map(list -> list.get(index))//
					.toList();
		};
		Function<VisualDatasetConf, SeriesDefinition> seriesFactory = datasetConf3 -> {
			return new SeriesDefinition(//
					valuesExtractor.apply(datasetConf3, visualConf.xIndex()), //
					valuesExtractor.apply(datasetConf3, visualConf.yIndex()), //
					datasetConf3.label());
		};
		Map<XYSeries, Color> coloredSeries = visualConf.datasetConfs().stream()//
				.collect(toMap(//
						datasetConf2 -> createSeries(seriesFactory.apply(datasetConf2)), //
						datasetConf -> datasetConf.color()//
				));
		XYSeriesCollection chartDataset = new XYSeriesCollection();
		coloredSeries.keySet().forEach(chartDataset::addSeries);

		JFreeChart chart = ChartFactory.createScatterPlot("MLP", // Title
				"x", // X-axis label
				"y", // Y-axis label
				chartDataset, // Dataset
				PlotOrientation.VERTICAL, // Orientation
				true, // Include legend
				true, // Tooltips
				false // URLs
		);

		XYPlot plot = (XYPlot) chart.getPlot();

		XYItemRenderer renderer = plot.getRenderer();
		coloredSeries.forEach((series, color) -> {
			int seriesIndex = chartDataset.getSeriesIndex(series.getKey());
			renderer.setSeriesPaint(seriesIndex, color);
		});

		addContour(visualConf.contourConf(), visualConf, plot);

		Consumer<List<RoundResult>> visualUpdater = _ -> {
			chart.fireChartChanged();
		};

		ChartPanel visualPanel = new ChartPanel(chart);

		return new Parts(visualPanel, visualUpdater);
	}

	private static Parts createLossPlot(LossPlotConf lossPlotConf) {
		XYSeriesCollection chartDataset = new XYSeriesCollection();
		XYSeries chartSeries = new XYSeries("Loss");
		chartDataset.addSeries(chartSeries);
		JFreeChart chart = ChartFactory.createXYLineChart(null, // Title
				"Rounds", // X-axis
				"Loss", // Y-axis
				chartDataset, // Dataset
				PlotOrientation.VERTICAL, // Orientation
				false, // Include legend
				true, // Tooltips
				false // URLs
		);
		// We often start with big numbers and end with small numbers
		// Use log scale to better show everything
		XYPlot xyPlot = chart.getXYPlot();
		ValueAxis initialAxis = xyPlot.getRangeAxis();
		xyPlot.setRangeAxis(new LogarithmicAxis(initialAxis.getLabel()));

		PlotUtils.Window<Long, Double> window = lossPlotConf.windowFactory()
				.createLongDoubleWindow((round, loss) -> chartSeries.add(round, loss));
		Consumer<List<RoundResult>> lossPlotUpdater = roundResults -> {
			roundResults.forEach(roundResult -> {
				long round = roundResult.round();
				double loss = roundResult.data().loss().data().get();
				window.feedWindow(round, loss);
			});
			chart.fireChartChanged();
		};

		ChartPanel chartPanel = new ChartPanel(chart);

		return new Parts(chartPanel, lossPlotUpdater);
	}

	private static Parts createTimePlot(TimePlotConf timePlotConf) {
		XYSeriesCollection chartDataset = new XYSeriesCollection();
		XYSeries computeSeries = new XYSeries("Compute");
		XYSeries backwardSeries = new XYSeries("Backward");
		XYSeries updateSeries = new XYSeries("Update");
		chartDataset.addSeries(computeSeries);
		chartDataset.addSeries(backwardSeries);
		chartDataset.addSeries(updateSeries);
		JFreeChart chart = ChartFactory.createXYLineChart(null, // Title
				"Rounds", // X-axis
				"Time (ns)", // Y-axis
				chartDataset, // Dataset
				PlotOrientation.VERTICAL, // Orientation
				true, // Include legend
				true, // Tooltips
				false // URLs
		);
		// We often start with big numbers and end with small numbers
		// Use log scale to better show everything
		XYPlot plot = chart.getXYPlot();
		ValueAxis initialAxis = plot.getRangeAxis();
		plot.setRangeAxis(new LogarithmicAxis(initialAxis.getLabel()));

		// To place the legend at the top
		chart.getLegend().setPosition(timePlotConf.legendPosition());

		PlotUtils.WindowFactory windowFactory = timePlotConf.windowFactory();

		PlotUtils.Window<Long, Long> computeWindow = windowFactory
				.createLongLongWindow((round, nanos) -> computeSeries.add(round, nanos));
		PlotUtils.Window<Long, Long> backwardWindow = windowFactory
				.createLongLongWindow((round, nanos) -> backwardSeries.add(round, nanos));
		PlotUtils.Window<Long, Long> updateWindow = windowFactory
				.createLongLongWindow((round, nanos) -> updateSeries.add(round, nanos));
		Function<RoundResult, BiConsumer<PlotUtils.Window<Long, Long>, Function<RoundData, Duration>>> seriesUpdaterFactory = roundResult -> {
			return (seriesWindow, durationExtractor) -> {
				long round = roundResult.round();
				long nanos = durationExtractor.apply(roundResult.data()).toNanos();
				seriesWindow.feedWindow(round, nanos);
			};
		};
		Consumer<List<RoundResult>> lossPlotUpdater = roundResults -> {
			roundResults.forEach(roundResult -> {
				var seriesUpdater = seriesUpdaterFactory.apply(roundResult);
				seriesUpdater.accept(computeWindow, RoundData::computeDuration);
				seriesUpdater.accept(backwardWindow, RoundData::backwardDuration);
				seriesUpdater.accept(updateWindow, RoundData::updateDuration);
			});
			chart.fireChartChanged();
		};

		ChartPanel chartPanel = new ChartPanel(chart);

		return new Parts(chartPanel, lossPlotUpdater);
	}

	private static JPanel createTrainPanel(TrainConf trainConf, Supplier<RoundData> mlpRound,
			Consumer<List<RoundResult>> roundConsumer) {
		JTextField roundsLimitField = FieldBuilder.buildFieldFor(trainConf.roundsLimit())//
				.intoText(src -> src.get().map(Object::toString).orElse(""))//
				.as(Long::parseLong).ifIs(value -> value > 0).thenApply((src, value) -> src.set(Optional.of(value)))//
				.whenEmptyApply(src -> src.set(Optional.empty()))//
				.otherwiseShow(FieldBuilder::error)//
				.build();

		JTextField batchSizeField = FieldBuilder.buildFieldFor(trainConf.batchSize())//
				.intoText(src -> Long.toString(src.get()))//
				.as(Long::parseLong).ifIs(value -> value > 0).thenApply(AtomicLong::set)//
				.otherwiseShow(FieldBuilder::error)//
				.build();

		JTextField updateStepField = FieldBuilder.buildFieldFor(trainConf.updateStep())//
				.intoText(src -> Double.toString(src.get()))//
				.as(Double::parseDouble).ifIs(value -> value > 0).thenApply(AtomicReference::set)//
				.otherwiseShow(FieldBuilder::error)//
				.build();

		JTextField roundField = new JTextField();
		JTextField lossField = new JTextField();
		Consumer<List<RoundResult>> lossFieldUpdater = roundResults -> {
			roundResults.forEach(roundResult -> {
				roundField.setText(Long.toString(roundResult.round()));
				lossField.setText(roundResult.data().loss().data().get().toString());
			});
		};

		JToggleButton trainButton = new JToggleButton(createTrainAction(trainConf, mlpRound, roundConsumer));
		roundConsumer = lossFieldUpdater.andThen(roundConsumer);

		JPanel trainPanel = new JPanel();
		trainPanel.setLayout(new GridBagLayout());
		{
			GridBagConstraints constraints = new GridBagConstraints();
			constraints.gridx = GridBagConstraints.RELATIVE;
			constraints.gridy = 0;
			constraints.insets = new Insets(5, 5, 5, 5);
			constraints.weighty = 0.0;
			constraints.fill = GridBagConstraints.HORIZONTAL;
			constraints.weightx = 1.0;
			trainPanel.add(trainButton, constraints);
			constraints.weightx = 1.0;
			trainPanel.add(roundsLimitField, constraints);
			constraints.weightx = 0.0;
			trainPanel.add(new JLabel("rounds by batch of"), constraints);
			constraints.weightx = 1.0;
			trainPanel.add(batchSizeField, constraints);
			constraints.weightx = 0.0;
			trainPanel.add(new JLabel("and step of"), constraints);
			constraints.weightx = 1.0;
			trainPanel.add(updateStepField, constraints);
			constraints.weightx = 0.0;
			trainPanel.add(new JLabel("|"), constraints);
			constraints.weightx = 0.0;
			trainPanel.add(new JLabel("Round:"), constraints);
			constraints.weightx = 1.0;
			trainPanel.add(roundField, constraints);
			constraints.weightx = 0.0;
			trainPanel.add(new JLabel("Loss:"), constraints);
			constraints.weightx = 3.0;// Bigger because usually more digits
			trainPanel.add(lossField, constraints);
		}
		return trainPanel;
	}

	private static void createMlpDot(NeuralNetConf neuralNetConf, MLP mlp, Path dotPath,
			Function<Input, String> inputIdSupplier, Function<Neuron, String> neuronIdSupplier,
			Function<Layer, String> layerIdSupplier) {
		Map<Object, String> ids = new HashMap<>();
		File dotFile = dotPath.toFile();
		try (PrintWriter dotWriter = new PrintWriter(dotFile)) {
			dotWriter.println("digraph G {");
			dotWriter.println("rankdir=LR");

			// Assume MLP, so all neurons of first layer have same inputs
			int inputsSize = mlp.layer(0).neuron(0).weights().size();

			// Manage inter-layers linking, since the info is not available from the MLP
			// TODO get input info from neuron to be agnostic of neural net architecture
			List<Object> previousLayer = new LinkedList<>();
			List<Object> currentLayer = new LinkedList<>();

			// Insert input layers with custom logics, since no object exists for them in
			// the MLP
			// TODO get input info from neuron to be agnostic of neural net architecture
			// TODO reuse non-input loop?
			String inputLayerId = layerIdSupplier.apply(null);
			dotWriter.println("subgraph cluster_" + inputLayerId + " {");
			dotWriter.println("color=" + dotColorCoder(neuralNetConf.clusterColor()) + ";");
			dotWriter.println("label = \"" + inputLayerId + "\";");
			IntStream.range(0, inputsSize).mapToObj(Input::new).forEach(input -> {
				currentLayer.add(input);
				String inputId = inputIdSupplier.apply(input);
				ids.put(input, inputId);
				dotWriter.println(inputId + " [label=\"" + inputId + "\", shape=square];");
			});
			dotWriter.println("}");

			// Insert non-input layers
			mlp.layers().forEach(layer -> {
				previousLayer.clear();
				previousLayer.addAll(currentLayer);
				currentLayer.clear();

				String layerId = layerIdSupplier.apply(layer);
				// TODO identify clusters from neurons with common inputs
				dotWriter.println("subgraph cluster_" + layerId + " {");
				dotWriter.println("color=" + dotColorCoder(neuralNetConf.clusterColor()) + ";");
				dotWriter.println("label = \"" + layerId + "\";");
				layer.neurons().forEach(neuron -> {
					currentLayer.add(neuron);
					String neuronId = neuronIdSupplier.apply(neuron);
					ids.put(neuron, neuronId);
					double bias = neuron.bias().data().get();
					dotWriter.println(
							neuronId + " [label=\"{{" + neuronId + " | " + roundForDot(bias) + "}}\", shape=record];");
					previousLayer.forEach(input -> {
						double weight = neuron.weights().get(previousLayer.indexOf(input)).data().get();
						dotWriter.println(
								ids.get(input) + " -> " + neuronId + " [label=\"" + roundForDot(weight) + "\"];");
					});
				});
				dotWriter.println("}");
			});

			dotWriter.println("}");
		} catch (FileNotFoundException cause) {
			throw new RuntimeException(cause);
		}
	}

	private static String dotColorCoder(Color color) {
		return "\"#" //
				+ String.format("%2x", color.getRed()) //
				+ String.format("%2x", color.getGreen())//
				+ String.format("%2x", color.getBlue()) //
				+ "\"";
	}

	private static void dotToFile(Path dotPath, Path pngPath, String type) {
		String[] command = { "dot", "-T" + type, dotPath.toString(), "-o", pngPath.toString() };
		Process process;
		try {
			process = Runtime.getRuntime().exec(command);
		} catch (IOException cause) {
			throw new RuntimeException(cause);
		}
		StringWriter writer = new StringWriter();
		try {
			process.errorReader().transferTo(writer);
		} catch (IOException cause) {
			throw new RuntimeException(writer.toString());
		}
		String error = writer.toString();
		int result;
		try {
			result = process.waitFor();
		} catch (InterruptedException cause) {
			throw new RuntimeException(cause);
		}
		if (result != 0) {
			throw new RuntimeException(error);
		}
	}

	private static <T> Function<T, String> prefixedCounterId(String prefix) {
		AtomicInteger counter = new AtomicInteger();
		return _ -> prefix + counter.getAndIncrement();
	}

	private static Stream<Node> streamOf(NodeList nodeList) {
		return IntStream.range(0, nodeList.getLength()).mapToObj(nodeList::item);
	}

	private static String titleOf(Node g) {
		return findFirstChild(g, "title").map(Node::getTextContent).orElse("∅");
	}

	private static Optional<Node> findFirstChild(Node g, String tag) {
		return streamOf(g.getChildNodes()).filter(child -> child.getNodeName().equals(tag)).findFirst();
	}

	private static Optional<Node> findSecondChild(Node g, String tag) {
		return streamOf(g.getChildNodes()).filter(child -> child.getNodeName().equals(tag)).skip(1).findFirst();
	}

	private static Path createTempPath(String fileName, String ext) {
		try {
			return Files.createTempFile(fileName, "." + ext);
		} catch (IOException cause) {
			throw new RuntimeException(cause);
		}
	}

	private static XYSeries createSeries(SeriesDefinition definition) {
		double[] xValues = toArray(definition.xs());
		double[] yValues = toArray(definition.ys());
		XYSeries series = new XYSeries(definition.lineTitle());
		for (int i = 0; i < xValues.length; i++) {
			series.add(xValues[i], yValues[i]);
		}
		return series;
	}

	private static double[] toArray(List<Double> xs) {
		return xs.stream().mapToDouble(d -> d).toArray();
	}

	private static void addContour(ContourConf contourConf, VisualConf visualConf, XYPlot plot) {
		Function<Double, Paint> contourPainter = createContourPainter(contourConf, visualConf);
		ContourDimension dimX = createContourDimension(plot.getDomainAxis(), contourConf.resolution().x());
		ContourDimension dimY = createContourDimension(plot.getRangeAxis(), contourConf.resolution().y());
		XYBlockRenderer contourRenderer = createContourRenderer(contourConf, contourPainter, dimX, dimY);
		XYZDataset contourDataset = createContourDataset(contourConf, dimX, dimY);

		int newIndex = plot.getDatasetCount();
		plot.setDataset(newIndex, contourDataset);
		plot.setRenderer(newIndex, contourRenderer);
	}

	private static Function<Double, Paint> createContourPainter(ContourConf contourConf, VisualConf visualConf) {
		return value -> {
			Color datasetColor = visualConf.datasetConfs().stream()//
					.filter(datasetConf -> datasetConf.predicate().test(value))//
					.findAny()//
					.map(VisualDatasetConf::color).orElse(visualConf.defaultColor());
			return contourConf.colorTransformation().apply(datasetColor, value);
		};
	}

	private static XYBlockRenderer createContourRenderer(ContourConf contourConf,
			Function<Double, Paint> contourPainter, ContourDimension dimX, ContourDimension dimY) {
		XYBlockRenderer contourRenderer = new XYBlockRenderer();
		contourRenderer.setBlockAnchor(RectangleAnchor.CENTER);
		contourRenderer.setBlockWidth(dimX.step());
		contourRenderer.setBlockHeight(dimY.step());
		contourRenderer.setPaintScale(new LookupPaintScale() {
			@Override
			public Paint getPaint(double value) {
				return contourPainter.apply(value);
			}
		});
		return contourRenderer;
	}

	private static ContourDimension createContourDimension(ValueAxis domainAxis, int resolutionX) {
		double minX = domainAxis.getLowerBound();
		double maxX = domainAxis.getUpperBound();
		return new ContourDimension(minX, maxX, resolutionX);
	}

	private static XYZDataset createContourDataset(ContourConf contourConf, ContourDimension dimX,
			ContourDimension dimY) {
		return new XYZDataset() {
			@Override
			public int getSeriesCount() {
				return 1;
			}

			@Override
			public int getItemCount(int series) {
				return dimX.resolution() * dimY.resolution();
			}

			@Override
			public double getXValue(int series, int item) {
				int xItem = item % dimX.resolution();
				double xUnit = (double) xItem / dimX.resolution();
				return xUnit * (dimX.max() - dimX.min()) + dimX.min();
			}

			@Override
			public double getYValue(int series, int item) {
				int yItem = (item / dimX.resolution()) % dimY.resolution();
				double yUnit = (double) yItem / dimX.resolution();
				return yUnit * (dimY.max() - dimY.min()) + dimY.min();
			}

			@Override
			public double getZValue(int series, int item) {
				double x = getXValue(series, item);
				double y = getYValue(series, item);
				return contourConf.function().apply(x, y);
			}

			@Override
			public Number getX(int series, int item) {
				return getXValue(series, item);
			}

			@Override
			public Number getY(int series, int item) {
				return getYValue(series, item);
			}

			@Override
			public Number getZ(int series, int item) {
				return getZValue(series, item);
			}

			@Override
			public void addChangeListener(DatasetChangeListener listener) {
				// ignore - this dataset never changes
			}

			@Override
			public void removeChangeListener(DatasetChangeListener listener) {
				// ignore
			}

			@Override
			public DatasetGroup getGroup() {
				return null;
			}

			@Override
			public void setGroup(DatasetGroup group) {
				// ignore
			}

			@Override
			public Comparable<?> getSeriesKey(int series) {
				return contourConf.name();
			}

			@Override
			public int indexOf(@SuppressWarnings("rawtypes") Comparable seriesKey) {
				return 0;
			}

			@Override
			public DomainOrder getDomainOrder() {
				return DomainOrder.ASCENDING;
			}
		};
	}

	private static AbstractAction createTrainAction(TrainConf trainConf, Supplier<RoundData> mlpRound,
			Consumer<List<RoundResult>> roundConsumer) {
		return new AbstractAction("Train") {
			long round = 0;

			@Override
			public void actionPerformed(ActionEvent e) {
				JToggleButton trainButton = (JToggleButton) e.getSource();
				if (!trainButton.isSelected()) {
					// Wait for the stuff to stop
				} else {
					AtomicReference<Optional<Long>> roundsLimit = trainConf.roundsLimit();
					AtomicLong batchSize = trainConf.batchSize();
					SwingUtilities.invokeLater(new Runnable() {
						@Override
						public void run() {
							if (isTrainingStopped()) {
								sendDataAndClean();
							} else if (isLimitReached()) {
								sendDataAndClean();
								stopTraining();
							} else if (isBatchFinished()) {
								sendDataAndClean();
								continueTraining();
							} else {
								computeRound();
								continueTraining();
							}
						}

						private void continueTraining() {
							SwingUtilities.invokeLater(this);
						}

						private void stopTraining() {
							trainButton.setSelected(false);
						}

						private boolean isTrainingStopped() {
							return !trainButton.isSelected();
						}

						private int batchedRound = 0;
						private final List<RoundResult> results = new LinkedList<>();

						private void computeRound() {
							batchedRound++;
							round++;
							RoundData data = mlpRound.get();
							results.add(new RoundResult(round, data));
						}

						private void sendDataAndClean() {
							roundConsumer.accept(results);
							results.clear();
							batchedRound = 0;
						}

						private boolean isBatchFinished() {
							long totalLimit = roundsLimit.get().orElse(Long.MAX_VALUE);
							long batchLimit = batchSize.get();
							long roundAtEndOfBatch = round - batchedRound + batchLimit;
							return round >= Math.min(totalLimit, roundAtEndOfBatch);
						}

						private boolean isLimitReached() {
							return roundsLimit.get().isPresent() && roundsLimit.get().get() <= round;
						}
					});
				}
			}
		};
	}

	private static double roundForDot(double value) {
		return (double) Math.round(value * 10000) / 10000;
	}

	private static void createCalculationDot(Value value, Path dotPath) {
		File dotFile = dotPath.toFile();
		AtomicInteger counter = new AtomicInteger();
		Function<Value, String> valueIdSupplier = LambdaUtils.memoize((_) -> "N" + counter.getAndIncrement());
		try (PrintWriter dotWriter = new PrintWriter(dotFile)) {
			dotWriter.println("digraph G {");
			dotWriter.println("rankdir=LR");// Left to right
			writeValueRecursively(value, dotWriter, valueIdSupplier, new HashSet<String>());
			dotWriter.println("}");
		} catch (FileNotFoundException cause) {
			throw new RuntimeException(cause);
		}
	}

	private static String writeValueRecursively(Value value, PrintWriter dotWriter,
			Function<Value, String> valueIdSupplier, Set<String> ids) {
		String id = valueIdSupplier.apply(value);
		if (ids.contains(id)) {
			// already inserted
		} else {
			dotWriter.println(id + " [label=\"{" + value.label() + " | {data " + formatData(value.data()) + " | grad "
					+ formatData(value.gradient()) + "}}\",shape=record];");
			Operator operator = value.operator();
			String operatorId = "op" + id;
			if (value.operator() != Operator.NONE) {
				dotWriter.println(operatorId + " [label=\"" + operator + "\"];");
				dotWriter.println(operatorId + " -> " + id + ";");
				value.operands().forEach(operand -> {
					String childId = writeValueRecursively(operand, dotWriter, valueIdSupplier, ids);
					dotWriter.println(childId + " -> " + operatorId + ";");
				});
			}
			ids.add(id);
		}
		return id;
	}

	private static String formatData(AtomicReference<Double> data) {
		return Optional.ofNullable(data.get()).map(App::roundForDot).map(Object::toString).orElse("∅");
	}

	// TODO Make private
	public record Resolution(int x, int y) {
	}

	record ContourDimension(double min, double max, int resolution) {
		public double step() {
			return (max() - min()) / resolution();
		}
	}

	// TODO Make private
	public record RoundData(Value loss, Duration computeDuration, Duration backwardDuration, Duration updateDuration) {
	}

	record RoundResult(long round, RoundData data) {
	}

	// TODO Make private
	public record VisualDatasetConf(//
			String label, //
			Predicate<Double> predicate, //
			Color color//
	) {
	}

	// TODO Make private
	public record VisualConf(//
			int xIndex, //
			int yIndex, //
			Color defaultColor, //
			Collection<VisualDatasetConf> datasetConfs, //
			ContourConf contourConf//
	) {
	}

	// TODO Make private
	public record ContourConf(//
			String name, //
			Resolution resolution, //
			BiFunction<Color, Double, Color> colorTransformation, //
			BinaryOperator<Double> function //
	) {
	}

	// TODO Make private
	public record TimePlotConf(PlotUtils.WindowFactory windowFactory, RectangleEdge legendPosition) {
	}

	// TODO Make private
	public record LossPlotConf(PlotUtils.WindowFactory windowFactory) {
	}

	// TODO Make private
	public record NeuralNetConf(int displayedDecimals, Color clusterColor) {
	}

	// TODO Make private
	public record Conf(//
			TrainConf trainConf, //
			NeuralNetConf neuralNetConf, //
			VisualConf visualConf, //
			LossPlotConf lossPlotConf, //
			TimePlotConf timePlotConf//
	) {
	}

	private record SeriesDefinition(List<Double> xs, List<Double> ys, String lineTitle) {
	}

	private record Parts(JPanel panel, Consumer<List<RoundResult>> panelUpdater) {
	}

	// TODO Make private
	public record TrainConf(//
			AtomicReference<Optional<Long>> roundsLimit, //
			AtomicLong batchSize, //
			AtomicReference<Double> updateStep//
	) {
	}
}