package fr.vergne.ai.swing;

import static java.util.stream.Collectors.toMap;

import java.awt.Color;
import java.awt.Component;
import java.awt.FlowLayout;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
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
import java.time.Instant;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Random;
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
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
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

import fr.vergne.ai.neuralnet.Dataset;
import fr.vergne.ai.neuralnet.NeuralNet.InputsBrowser;
import fr.vergne.ai.neuralnet.NeuralNet.MLP;
import fr.vergne.ai.neuralnet.NeuralNet.NeuralNetBrowser;
import fr.vergne.ai.neuralnet.NeuralNet.Operator;
import fr.vergne.ai.neuralnet.NeuralNet.ParameterNamer;
import fr.vergne.ai.neuralnet.NeuralNet.Value;
import fr.vergne.ai.utils.LambdaUtils;

@SuppressWarnings("serial")
public class App extends JFrame {

	public App() {
		State state;
		{
			Random random = new Random(0);

			AtomicReference<Optional<Long>> roundsLimit = new AtomicReference<>(Optional.empty());
			AtomicLong batchSize = new AtomicLong(1);
			AtomicReference<Double> updateStep = new AtomicReference<Double>(0.001);
			TrainState trainState = new TrainState(roundsLimit, batchSize, updateStep);

			Collection<VisualDatasetState> datasetStates = List.of(//
					new VisualDatasetState("1.0", value -> value > 0, Color.RED), //
					new VisualDatasetState("-1.0", value -> value < 0, Color.BLUE)//
			);

			Resolution contourResolution = new Resolution(100, 100);
			BiFunction<Color, Double, Color> colorTransformation = (color, value) -> adaptSaturation(color,
					Math.abs(value) * 0.6);
			Function<MLP, BinaryOperator<Double>> contourFunctionFactory = mlp -> (x, y) -> {
				return mlp.computeRaw(List.of(x, y)).get(0);
			};
			ContourState contourState = new ContourState("MLP", contourResolution, colorTransformation,
					contourFunctionFactory);

			int xIndex = 0;
			int yIndex = 1;
			Color defaultColor = Color.BLACK;// transparent
			VisualState visualState = new VisualState(xIndex, yIndex, defaultColor, datasetStates, contourState);

			PlotUtils.WindowFactory windowFactory = switch (2) {
			case 1 -> PlotUtils.createNoWindowFactory();
			case 2 -> PlotUtils.createFixedWindowFactory(100);
			case 3 -> PlotUtils.createSlidingWindowFactory(10);
			default -> throw new IllegalArgumentException("Unexpected window factory");
			};
			RectangleEdge legendPosition = RectangleEdge.TOP;
			TimePlotState timePlotState = new TimePlotState(windowFactory, legendPosition);

			LossPlotState lossPlotState = new LossPlotState(windowFactory);

			int displayedDecimals = 4;
			Color clusterColor = Color.LIGHT_GRAY;
			NeuralNetState neuralNetState = new NeuralNetState(random, displayedDecimals, clusterColor);

			DatasetState datasetState = new DatasetState(random);

			state = new State(trainState, datasetState, neuralNetState, visualState, lossPlotState, timePlotState);
		}

		Parts datasetParts = createDatasetPanel(state.datasetState());
		Parts mlpParts = createMlpPanel(state.neuralNetState());
		Parts visualParts = createVisual(state.visualState(), datasetParts.trainDatasetNotifier(),
				mlpParts.mlpNotifier());
		Parts lossPlotParts = createLossPlot(state.lossPlotState());
		Parts timePlotParts = createTimePlot(state.timePlotState());

		JTabbedPane tabs = new JTabbedPane();
		tabs.add(datasetParts.panel(), "Dataset");
		tabs.add(mlpParts.panel(), "MLP");
		tabs.add(visualParts.panel(), "Visual");
		tabs.add(lossPlotParts.panel(), "Loss plot");
		tabs.add(timePlotParts.panel(), "Time plot");

		Consumer<List<RoundResult>> roundConsumer = datasetParts.panelUpdater()//
				.andThen(mlpParts.panelUpdater())//
				.andThen(visualParts.panelUpdater())//
				.andThen(lossPlotParts.panelUpdater())//
				.andThen(timePlotParts.panelUpdater())//
		;

		JPanel trainPanel = createTrainPanel(state.trainState(), roundConsumer, datasetParts.trainDatasetNotifier(),
				datasetParts.testDatasetNotifier(), mlpParts.mlpNotifier());

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

		this.setTitle("MLP");
	}

	private Parts createDatasetPanel(DatasetState datasetState) {
		record Entry(String label, Supplier<Map<List<Double>, Double>> trainDatasetSupplier,
				Supplier<Map<List<Double>, Double>> testDatasetSupplier) {
			public Entry(String label, Supplier<Map<List<Double>, Double>> datasetSupplier) {
				this(label, datasetSupplier, datasetSupplier);
			}
		}
		List<Entry> datasetCandidates = List.of(//
				new Entry("Circle", () -> Dataset.circle(datasetState.random())), //
				new Entry("Columns 1", () -> Dataset.columns1(datasetState.random())), //
				new Entry("Columns 2", () -> Dataset.columns2(datasetState.random())), //
				new Entry("Columns steep", () -> Dataset.steepColumns(datasetState.random())), //
				new Entry("Moons", () -> Dataset.moons(datasetState.random()))//
		);
		// TODO Implement dataset import (add to candidates + autoload)

		JPanel panel = new JPanel(new GridBagLayout());
		GridBagConstraints constraints = new GridBagConstraints();
		constraints.gridx = 0;
		constraints.gridy = GridBagConstraints.RELATIVE;
		constraints.anchor = GridBagConstraints.LINE_START;

		ButtonGroup group = new ButtonGroup();

		DataNotifier<Map<List<Double>, Double>> trainDatasetNotifier = new DataNotifier<>();
		DataNotifier<Map<List<Double>, Double>> testDatasetNotifier = new DataNotifier<>();
		datasetCandidates.forEach(entry -> {
			var label = entry.label();
			var trainDatasetSupplier = entry.trainDatasetSupplier();
			var testDatasetSupplier = entry.testDatasetSupplier();

			JRadioButton radioButton = new JRadioButton(label);
			radioButton.addActionListener(_ -> {
				trainDatasetNotifier.update(trainDatasetSupplier.get());
				testDatasetNotifier.update(testDatasetSupplier.get());
			});

			panel.add(radioButton, constraints);

			group.add(radioButton);
		});

		Consumer<List<RoundResult>> updater = _ -> {
			// Nothing to do with rounds results
		};

		return new Parts(panel, updater, null, trainDatasetNotifier, testDatasetNotifier);
	}

	private Parts createMlpPanel(NeuralNetState neuralNetState) {
		record Entry(String label, Supplier<MLP> mlpSupplier) {
			@Override
			public final String toString() {
				return label;
			}
		}
		List<Entry> mlpCandidates = List.of(//
				new Entry("4x1",
						() -> new MLP(ParameterNamer.create(), 2, List.of(4, 1),
								(_) -> neuralNetState.random().nextDouble(-1.0, 1.0))), //
				new Entry("20x1",
						() -> new MLP(ParameterNamer.create(), 2, List.of(20, 1),
								(_) -> neuralNetState.random().nextDouble(-1.0, 1.0))), //
				new Entry("4x4x4x4x1", () -> new MLP(ParameterNamer.create(), 2, List.of(4, 4, 4, 4, 1),
						(_) -> neuralNetState.random().nextDouble(-1.0, 1.0)))//
		);
		// TODO Implement MLP import (add to candidates + autoload)
		// TODO Implement MLP building

		var ctx = new Object() {
			NeuralNetBrowser neuralNetBrowser;
		};
		DataNotifier<MLP> mlpNotifier = new DataNotifier<MLP>();
		JComboBox<Entry> factoryComboBox = new JComboBox<>(mlpCandidates.toArray(new Entry[0]));
		JSVGCanvas svgCanvas = new JSVGCanvas();
		factoryComboBox.addActionListener(_ -> {
			Entry selectedOption = (Entry) factoryComboBox.getSelectedItem();
			MLP mlp = selectedOption.mlpSupplier().get();
			mlpNotifier.update(mlp);

			NeuralNetBrowser neuralNetBrowser = NeuralNetBrowser.forMlp(mlp, prefixedCounterId("I"),
					prefixedCounterId("N"), prefixedCounterId("L"));

			String fileName = "graph";
			Path dotPath = createTempPath(fileName, "dot");
			createMlpDot(neuralNetState, () -> neuralNetBrowser, dotPath);

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

			svgCanvas.setDocumentState(JSVGCanvas.ALWAYS_DYNAMIC);
			svgCanvas.setSVGDocument(createdDocument);
			ctx.neuralNetBrowser = neuralNetBrowser;
			mlpNotifier.update(mlp);
		});

		Consumer<List<RoundResult>> mlpUpdater = _ -> {
			// A deep copy might be done, so retrieve actually stored instance
			SVGDocument svgDocument = svgCanvas.getSVGDocument();
			// We will work on it from its root
			SVGSVGElement root = svgDocument.getRootElement();
			/*
			 * As per Batik documentation, for a dynamic SVG to properly trigger its
			 * repainting, the changes must be performed in the updater manager queue.
			 */
			svgCanvas.getUpdateManager().getUpdateRunnableQueue().invokeLater(() -> {
				int decimals = neuralNetState.displayedDecimals();
				Function<Double, String> parameterFormatter = value -> {
					// TODO Use app locale?
					return String.format(Locale.US, "%." + decimals + "f", value);
				};
				NeuralNetBrowser neuralNetBrowser = ctx.neuralNetBrowser;
				streamOf(root.getElementsByTagName("g")).forEach(groupNode -> {
					String title = titleOf(groupNode);
					if (title.matches("[IN][0-9]+->N[0-9]+")) {
						int separatorStart = title.indexOf("->");
						int separatorEnd = separatorStart + 2;
						String inputId = title.substring(0, separatorStart);
						String outputId = title.substring(separatorEnd);
						double weight = neuralNetBrowser.neuron(outputId).weightWith(inputId);
						Node weightNode = findFirstChild(groupNode, "text").orElseThrow();
						weightNode.setTextContent(parameterFormatter.apply(weight));
					} else if (title.matches("N[0-9]+")) {
						String neuronId = title;
						double bias = neuralNetBrowser.neuron(neuronId).bias();
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

		Supplier<NeuralNetBrowser> neuralNetBrowserSupplier = () -> ctx.neuralNetBrowser;
		JPanel exportPanel = createExportPanel(neuralNetState, neuralNetBrowserSupplier);

		JPanel mlpPanel = new JPanel();
		mlpPanel.setLayout(new GridBagLayout());
		GridBagConstraints constraints = new GridBagConstraints();
		constraints.gridx = 1;
		constraints.gridy = GridBagConstraints.RELATIVE;
		constraints.weightx = 1.0;
		constraints.fill = GridBagConstraints.BOTH;
		constraints.weighty = 0.0;
		mlpPanel.add(factoryComboBox, constraints);
		constraints.weighty = 1.0;
		mlpPanel.add(svgCanvas, constraints);
		constraints.weighty = 0.0;
		mlpPanel.add(exportPanel, constraints);

		return new Parts(mlpPanel, mlpUpdater, mlpNotifier, null, null);
	}

	private JPanel createExportPanel(NeuralNetState neuralNetState,
			Supplier<NeuralNetBrowser> neuralNetBrowserSupplier) {
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
						createMlpDot(neuralNetState, neuralNetBrowserSupplier, file.toPath());
					} else if (fileName.endsWith(".svg") || fileName.endsWith(".pdf") || fileName.endsWith(".png")) {
						if (existsAndOverrideRejected(file)) {
							return; // User chose not to overwrite, do nothing
						}
						Path tempDotPath = createTempPath(fileName, "dot");
						createMlpDot(neuralNetState, neuralNetBrowserSupplier, tempDotPath);
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

	private static Parts createVisual(VisualState visualState, DataNotifier<Map<List<Double>, Double>> datasetNotifier,
			DataNotifier<MLP> mlpNotifier) {
		JPanel visualPanel = new JPanel(new GridLayout(1, 1));

		var ctx = new Object() {
			Map<List<Double>, Double> dataset;
			MLP mlp;
		};

		Runnable repainter = () -> {
			if (ctx.mlp == null || ctx.dataset == null) {
				// Incomplete context, don't paint yet
				return;
			}

			BiFunction<VisualDatasetState, Integer, List<Double>> valuesExtractor = (datasetState, index) -> {
				return ctx.dataset.entrySet().stream()//
						.filter(entry -> datasetState.predicate().test(entry.getValue()))//
						.map(Entry::getKey)//
						.map(list -> list.get(index))//
						.toList();
			};
			Function<VisualDatasetState, SeriesDefinition> seriesFactory = datasetState -> {
				return new SeriesDefinition(//
						valuesExtractor.apply(datasetState, visualState.xIndex()), //
						valuesExtractor.apply(datasetState, visualState.yIndex()), //
						datasetState.label());
			};
			Map<XYSeries, Color> coloredSeries = visualState.datasetStates().stream()//
					.collect(toMap(//
							datasetState -> createSeries(seriesFactory.apply(datasetState)), //
							datasetState -> datasetState.color()//
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

			addContour(visualState.contourState(), visualState, plot, ctx.mlp);

			ChartPanel chartPanel = new ChartPanel(chart);

			visualPanel.removeAll();
			visualPanel.add(chartPanel);
		};

		// Update data upon notification
		datasetNotifier.addListener(dataset -> ctx.dataset = dataset);
		mlpNotifier.addListener(mlp -> ctx.mlp = mlp);

		// Repaint upon notification
		datasetNotifier.addListener(_ -> repainter.run());
		mlpNotifier.addListener(_ -> repainter.run());

		Consumer<List<RoundResult>> visualUpdater = _ -> {
			Component component = visualPanel.getComponent(0);
			if (component instanceof ChartPanel chartPanel) {
				chartPanel.getChart().fireChartChanged();
			}
		};

		return new Parts(visualPanel, visualUpdater, null, null, null);
	}

	private static Parts createLossPlot(LossPlotState lossPlotState) {
		XYSeriesCollection chartDataset = new XYSeriesCollection();
		XYSeries trainSeries = new XYSeries("Train");
		XYSeries testSeries = new XYSeries("Test");
		chartDataset.addSeries(trainSeries);
		chartDataset.addSeries(testSeries);
		JFreeChart chart = ChartFactory.createXYLineChart(null, // Title
				"Rounds", // X-axis
				null, // Y-axis
				chartDataset, // Dataset
				PlotOrientation.VERTICAL, // Orientation
				true, // Include legend
				true, // Tooltips
				false // URLs
		);
		// We often start with big numbers and end with small numbers
		// Use log scale to better show everything
		XYPlot xyPlot = chart.getXYPlot();
		ValueAxis initialAxis = xyPlot.getRangeAxis();
		xyPlot.setRangeAxis(new LogarithmicAxis(initialAxis.getLabel()));

		PlotUtils.Window<Long, Double> trainWindow = lossPlotState.windowFactory()
				.createLongDoubleWindow((round, loss) -> trainSeries.add(round, loss));
		PlotUtils.Window<Long, Double> testWindow = lossPlotState.windowFactory()
				.createLongDoubleWindow((round, loss) -> testSeries.add(round, loss));
		Consumer<List<RoundResult>> lossPlotUpdater = roundResults -> {
			roundResults.forEach(roundResult -> {
				long round = roundResult.round();
				trainWindow.feedWindow(round, roundResult.data().trainLoss());
				testWindow.feedWindow(round, roundResult.data().testLoss());
			});
			chart.fireChartChanged();
		};

		ChartPanel chartPanel = new ChartPanel(chart);

		return new Parts(chartPanel, lossPlotUpdater, null, null, null);
	}

	private static Parts createTimePlot(TimePlotState timePlotState) {
		XYSeriesCollection chartDataset = new XYSeriesCollection();
		XYSeries computeSeries = new XYSeries("Compute");
		XYSeries testSeries = new XYSeries("Test");
		XYSeries backwardSeries = new XYSeries("Backward");
		XYSeries updateSeries = new XYSeries("Update");
		chartDataset.addSeries(computeSeries);
		chartDataset.addSeries(testSeries);
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
		chart.getLegend().setPosition(timePlotState.legendPosition());

		PlotUtils.WindowFactory windowFactory = timePlotState.windowFactory();

		PlotUtils.Window<Long, Long> computeWindow = windowFactory
				.createLongLongWindow((round, nanos) -> computeSeries.add(round, nanos));
		PlotUtils.Window<Long, Long> testWindow = windowFactory
				.createLongLongWindow((round, nanos) -> testSeries.add(round, nanos));
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
				seriesUpdater.accept(testWindow, RoundData::testDuration);
				seriesUpdater.accept(backwardWindow, RoundData::backwardDuration);
				seriesUpdater.accept(updateWindow, RoundData::updateDuration);
			});
			chart.fireChartChanged();
		};

		ChartPanel chartPanel = new ChartPanel(chart);

		return new Parts(chartPanel, lossPlotUpdater, null, null, null);
	}

	private static JPanel createTrainPanel(TrainState trainState, Consumer<List<RoundResult>> roundsConsumer,
			DataNotifier<Map<List<Double>, Double>> trainDatasetNotifier,
			DataNotifier<Map<List<Double>, Double>> testDatasetNotifier, DataNotifier<MLP> mlpNotifier) {
		JTextField roundsLimitField = FieldBuilder.buildFieldFor(trainState.roundsLimit())//
				.intoText(src -> src.get().map(Object::toString).orElse(""))//
				.as(Long::parseLong).ifIs(value -> value > 0).thenApply((src, value) -> src.set(Optional.of(value)))//
				.whenEmptyApply(src -> src.set(Optional.empty()))//
				.otherwiseShow(FieldBuilder::error)//
				.build();

		JTextField batchSizeField = FieldBuilder.buildFieldFor(trainState.batchSize())//
				.intoText(src -> Long.toString(src.get()))//
				.as(Long::parseLong).ifIs(value -> value > 0).thenApply(AtomicLong::set)//
				.otherwiseShow(FieldBuilder::error)//
				.build();

		JTextField updateStepField = FieldBuilder.buildFieldFor(trainState.updateStep())//
				.intoText(src -> Double.toString(src.get()))//
				.as(Double::parseDouble).ifIs(value -> value > 0).thenApply(AtomicReference::set)//
				.otherwiseShow(FieldBuilder::error)//
				.build();

		JLabel roundLabel = new JLabel();
		JLabel trainLossLabel = new JLabel();
		JLabel testLossLabel = new JLabel();
		Consumer<List<RoundResult>> lossFieldUpdater = roundResults -> {
			roundResults.forEach(roundResult -> {
				roundLabel.setText(Long.toString(roundResult.round()));
				trainLossLabel.setText(scientificFormat(roundResult.data().trainLoss()));
				testLossLabel.setText(scientificFormat(roundResult.data().testLoss()));
			});
		};

		roundsConsumer = lossFieldUpdater.andThen(roundsConsumer);
		JToggleButton trainButton = new JToggleButton(
				createTrainAction(trainState, roundsConsumer, trainDatasetNotifier, testDatasetNotifier, mlpNotifier));

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
			/////////////////
			constraints.gridy++;
			constraints.weightx = 0.0;
			trainPanel.add(new JLabel("Round:", JLabel.TRAILING), constraints);
			constraints.weightx = 1.0;
			trainPanel.add(roundLabel, constraints);
			constraints.weightx = 0.0;
			trainPanel.add(new JLabel("Train loss:", JLabel.TRAILING), constraints);
			constraints.weightx = 1.0;
			trainPanel.add(trainLossLabel, constraints);
			constraints.weightx = 0.0;
			trainPanel.add(new JLabel("Test loss:", JLabel.TRAILING), constraints);
			constraints.weightx = 1.0;
			trainPanel.add(testLossLabel, constraints);
		}
		return trainPanel;
	}

	private static void createMlpDot(NeuralNetState neuralNetState, Supplier<NeuralNetBrowser> neuralNetBrowserSupplier,
			Path dotPath) {
		NeuralNetBrowser neuralNetBrowser = neuralNetBrowserSupplier.get();
		Map<Object, String> ids = new HashMap<>();
		File dotFile = dotPath.toFile();
		try (PrintWriter dotWriter = new PrintWriter(dotFile)) {
			dotWriter.println("digraph G {");
			dotWriter.println("rankdir=LR");

			InputsBrowser inputsCluster = neuralNetBrowser.inputsCluster();
			String inputsClusterId = inputsCluster.id().toString();
			dotWriter.println("subgraph cluster_" + inputsClusterId + " {");
			dotWriter.println("color=" + dotColorCoder(neuralNetState.clusterColor()) + ";");
			dotWriter.println("label = \"" + inputsClusterId + "\";");
			inputsCluster.inputs().forEach(input -> {
				String inputId = input.id().toString();
				ids.put(input, inputId);
				dotWriter.println(inputId + " [label=\"" + inputId + "\", shape=square];");
			});
			dotWriter.println("}");

			neuralNetBrowser.neuronsClusters().forEach(cluster -> {
				String clusterId = cluster.id().toString();
				dotWriter.println("subgraph cluster_" + clusterId + " {");
				dotWriter.println("color=" + dotColorCoder(neuralNetState.clusterColor()) + ";");
				dotWriter.println("label = \"" + clusterId + "\";");
				cluster.neurons().forEach(neuron -> {
					String neuronId = neuron.id().toString();
					ids.put(neuron, neuronId);
					double bias = neuron.bias();
					dotWriter.println(
							neuronId + " [label=\"{{" + neuronId + " | " + roundForDot(bias) + "}}\", shape=record];");
					neuron.weightedInputs().forEach(input -> {
						dotWriter.println(
								input.id() + " -> " + neuronId + " [label=\"" + roundForDot(input.weight()) + "\"];");
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

	private static <T> Supplier<String> prefixedCounterId(String prefix) {
		AtomicInteger counter = new AtomicInteger();
		return () -> prefix + counter.getAndIncrement();
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

	private static void addContour(ContourState contourState, VisualState visualState, XYPlot plot, MLP mlp) {
		Function<Double, Paint> contourPainter = createContourPainter(contourState, visualState);
		ContourDimension dimX = createContourDimension(plot.getDomainAxis(), contourState.resolution().x());
		ContourDimension dimY = createContourDimension(plot.getRangeAxis(), contourState.resolution().y());
		XYBlockRenderer contourRenderer = createContourRenderer(contourState, contourPainter, dimX, dimY);
		XYZDataset contourDataset = createContourDataset(contourState, dimX, dimY, mlp);

		int newIndex = plot.getDatasetCount();
		plot.setDataset(newIndex, contourDataset);
		plot.setRenderer(newIndex, contourRenderer);
	}

	private static Function<Double, Paint> createContourPainter(ContourState contourState, VisualState visualState) {
		return value -> {
			Color datasetColor = visualState.datasetStates().stream()//
					.filter(datasetState -> datasetState.predicate().test(value))//
					.findAny()//
					.map(VisualDatasetState::color).orElse(visualState.defaultColor());
			return contourState.colorTransformation().apply(datasetColor, value);
		};
	}

	private static XYBlockRenderer createContourRenderer(ContourState contourState,
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

	private static XYZDataset createContourDataset(ContourState contourState, ContourDimension dimX,
			ContourDimension dimY, MLP mlp) {
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
				return contourState.functionFactory().apply(mlp).apply(x, y);
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
				return contourState.name();
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

	private static AbstractAction createTrainAction(TrainState trainState, Consumer<List<RoundResult>> roundsConsumer,
			DataNotifier<Map<List<Double>, Double>> trainDatasetNotifier,
			DataNotifier<Map<List<Double>, Double>> testDatasetNotifier, DataNotifier<MLP> mlpNotifier) {

		var ctx = new Object() {
			Map<List<Double>, Double> trainDataset = null;
			Map<List<Double>, Double> testDataset = null;
			MLP mlp = null;
		};

		// Update data upon notification
		trainDatasetNotifier.addListener(dataset -> ctx.trainDataset = dataset);
		testDatasetNotifier.addListener(dataset -> ctx.testDataset = dataset);
		mlpNotifier.addListener(mlp -> ctx.mlp = mlp);

		return new AbstractAction("Train") {
			long round = 0;

			@Override
			public void actionPerformed(ActionEvent e) {
				JToggleButton trainButton = (JToggleButton) e.getSource();
				if (!trainButton.isSelected()) {
					// Wait for the stuff to stop
				} else {
					AtomicReference<Optional<Long>> roundsLimit = trainState.roundsLimit();
					AtomicLong batchSize = trainState.batchSize();
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
							MLP mlp = ctx.mlp;
							Map<List<Double>, Double> trainDataset = ctx.trainDataset;
							Map<List<Double>, Double> testDataset = ctx.testDataset;
							Instant startInstant = Instant.now();
							Value trainLoss = mlp.computeLoss(trainDataset);
							Instant computedInstant = Instant.now();
							Value testLoss = mlp.computeLoss(testDataset);
							Instant testedInstant = Instant.now();
							trainLoss.backward();
							Instant backwardedInstant = Instant.now();
							mlp.updateParameters(trainState.updateStep().get());
							Instant updatedInstant = Instant.now();
							RoundData data = new RoundData(//
									trainLoss.data().get(), //
									testLoss.data().get(), //
									Duration.between(startInstant, computedInstant), //
									Duration.between(computedInstant, testedInstant), //
									Duration.between(testedInstant, backwardedInstant), //
									Duration.between(backwardedInstant, updatedInstant)//
							);
							results.add(new RoundResult(round, data));
						}

						private void sendDataAndClean() {
							roundsConsumer.accept(results);
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

	private Color adaptSaturation(Color color, double factor) {
		float[] hsb = Color.RGBtoHSB(color.getRed(), color.getGreen(), color.getBlue(), null);
		hsb[1] *= factor;
		return Color.getHSBColor(hsb[0], hsb[1], hsb[2]);
	}

	record Resolution(int x, int y) {
	}

	record ContourDimension(double min, double max, int resolution) {
		public double step() {
			return (max() - min()) / resolution();
		}
	}

	record RoundData(double trainLoss, double testLoss, Duration computeDuration, Duration testDuration,
			Duration backwardDuration, Duration updateDuration) {
	}

	record RoundResult(long round, RoundData data) {
	}

	record VisualDatasetState(//
			String label, //
			Predicate<Double> predicate, //
			Color color//
	) {
	}

	record VisualState(//
			int xIndex, //
			int yIndex, //
			Color defaultColor, //
			Collection<VisualDatasetState> datasetStates, //
			ContourState contourState//
	) {
	}

	record ContourState(//
			String name, //
			Resolution resolution, //
			BiFunction<Color, Double, Color> colorTransformation, //
			Function<MLP, BinaryOperator<Double>> functionFactory //
	) {
	}

	record TimePlotState(PlotUtils.WindowFactory windowFactory, RectangleEdge legendPosition) {
	}

	record LossPlotState(PlotUtils.WindowFactory windowFactory) {
	}

	record NeuralNetState(//
			Random random, //
			int displayedDecimals, //
			Color clusterColor//
	) {
	}

	record DatasetState(Random random) {
	}

	record State(//
			TrainState trainState, //
			DatasetState datasetState, //
			NeuralNetState neuralNetState, //
			VisualState visualState, //
			LossPlotState lossPlotState, //
			TimePlotState timePlotState//
	) {
	}

	private record SeriesDefinition(List<Double> xs, List<Double> ys, String lineTitle) {
	}

	private record Parts(JPanel panel, Consumer<List<RoundResult>> panelUpdater, DataNotifier<MLP> mlpNotifier,
			DataNotifier<Map<List<Double>, Double>> trainDatasetNotifier,
			DataNotifier<Map<List<Double>, Double>> testDatasetNotifier) {
	}

	record TrainState(//
			AtomicReference<Optional<Long>> roundsLimit, //
			AtomicLong batchSize, //
			AtomicReference<Double> updateStep//
	) {
	}

	static class DataNotifier<T> {
		private final List<DataNotifier.Listener<T>> listeners = new LinkedList<>();

		void addListener(DataNotifier.Listener<T> listener) {
			listeners.add(listener);
		}

		void update(T newData) {
			this.listeners.forEach(listener -> listener.updated(newData));
		}

		interface Listener<T> {
			void updated(T newData);
		}
	}

	private static String scientificFormat(double value) {
		return String.format("%.2e", value);
	}
}