package fr.vergne.ai.neuralnet;

import static java.util.stream.Collectors.toMap;

import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.Paint;
import java.awt.event.ActionEvent;
import java.awt.image.BufferedImage;
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
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
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

import javax.imageio.ImageIO;
import javax.swing.AbstractAction;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.JTextField;
import javax.swing.JToggleButton;
import javax.swing.SwingUtilities;

import org.apache.batik.anim.dom.SAXSVGDocumentFactory;
import org.apache.batik.bridge.UpdateManager;
import org.apache.batik.bridge.UpdateManagerEvent;
import org.apache.batik.bridge.UpdateManagerListener;
import org.apache.batik.swing.JSVGCanvas;
import org.apache.batik.swing.gvt.GVTTreeRendererEvent;
import org.apache.batik.swing.gvt.GVTTreeRendererListener;
import org.apache.batik.swing.svg.GVTTreeBuilderEvent;
import org.apache.batik.swing.svg.GVTTreeBuilderListener;
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
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.svg.SVGDocument;
import org.w3c.dom.svg.SVGSVGElement;

public class NeuralNet {

	enum Operator {
		NONE(null, //
				(_) -> {
					throw new RuntimeException("Should not be called");
				}, //
				(_, operands) ->

				{
					restrict(operands, 0);// Nothing to resolve
				}//
		), //

		PLUS("+", //
				(operands) -> operands.stream().mapToDouble(d -> d).sum(), //
				(parentGradient, operands) -> {
					operands.forEach(operand -> operand.computeGradientsRecursivelyStartingHereAt(parentGradient));
				}//
		), //
		MINUS("-", //
				(operands) -> {
					restrict(operands, 2);
					return operands.get(0) - operands.get(1);
				}, //
				(parentGradient, operands) -> {
					operands.forEach(operand -> operand.computeGradientsRecursivelyStartingHereAt(parentGradient));
				}//
		), //
		MULT("*", //
				(operands) -> operands.stream().mapToDouble(d -> d).reduce((a, b) -> a * b).getAsDouble(), //
				(parentGradient, operands) -> {
					restrict(operands, 2);
					Value op1 = operands.get(0);
					Value op2 = operands.get(1);
					op1.computeGradientsRecursivelyStartingHereAt(parentGradient * op2.data().get());
					op2.computeGradientsRecursivelyStartingHereAt(parentGradient * op1.data().get());
				}//
		), //
		POW("^", //
				(operands) -> {
					restrict(operands, 2);
					return Math.pow(operands.get(0), operands.get(1));
				}, //
				(parentGradient, operands) -> {
					restrict(operands, 2);
					Double v1 = operands.get(0).data().get();
					Double v2 = operands.get(1).data().get();
					double gradient = v2 * Math.pow(v1, v2 - 1);
					operands.get(0).computeGradientsRecursivelyStartingHereAt(parentGradient * gradient);
				}//
		), //
		EXP("exp", //
				(operands) -> {
					restrict(operands, 1);
					return Math.exp(operands.get(0));
				}, //
				(parentGradient, operands) -> {
					restrict(operands, 1);
					double gradient = Math.exp(operands.get(0).data().get());
					operands.get(0).computeGradientsRecursivelyStartingHereAt(parentGradient * gradient);
				}//
		), //
		TANH("tanh", //
				(operands) -> {
					restrict(operands, 1);
					return Math.tanh(operands.get(0));
				}, //
				(parentGradient, operands) -> {
					restrict(operands, 1);
					double tanh = Math.tanh(operands.get(0).data().get());
					double gradient = 1 - tanh * tanh;
					operands.get(0).computeGradientsRecursivelyStartingHereAt(parentGradient * gradient);
				}//
		);

		private final String label;
		private final Function<List<Double>, Double> computer;
		private final BiConsumer<Double, List<Value>> gradientResolver;

		Operator(String label, Function<List<Double>, Double> computer,
				BiConsumer<Double, List<Value>> gradientResolver) {
			this.label = label;
			this.computer = computer;
			this.gradientResolver = gradientResolver;
		}

		public Double compute(List<Double> operands) {
			return computer.apply(operands);
		}

		public void resolveGradients(double grad, List<Value> operands) {
			gradientResolver.accept(grad, operands);
		}

		@Override
		public String toString() {
			return label;
		}
	}

	public static record Value(String label, AtomicReference<Double> data, Operator operator, List<Value> operands,
			AtomicReference<Double> gradient) {

		public Value(String label, double data) {
			this(label, new AtomicReference<>(data), Operator.NONE, Collections.emptyList(), new AtomicReference<>());
		}

		public Value(String label, Function<String, Double> dataInitializer) {
			this(label, dataInitializer.apply(label));
		}

		public Value(Supplier<String> label, Function<String, Double> dataInitializer) {
			this(label.get(), dataInitializer);
		}

		public Value updateDataRecursivelyWeigthedWithGradients(double step) {
			if (operator == Operator.NONE) {
				data.accumulateAndGet(step * gradient.get(), (t, u) -> t + u);
			} else {
				operands.forEach(operand -> operand.updateDataRecursivelyWeigthedWithGradients(step));
				Double computed = operator
						.compute(operands.stream().map(Value::data).map(AtomicReference::get).toList());
				data.set(computed);
			}
			return this;
		}

		public Value named(String label) {
			return new Value(label, data, operator, operands, gradient);
		}

		public void setGradient(double gradient) {
			this.gradient().set(gradient);
		}

		private void addGradient(double gradientDelta) {
			this.gradient().accumulateAndGet(gradientDelta, (g1, g2) -> g1 + g2);
		}

		public void resetGradientsRecursively() {
			setGradient(0.0);
			operands.forEach(Value::resetGradientsRecursively);
		}

		public void computeGradientsRecursivelyStartingHereAt(double gradient) {
			addGradient(gradient);
			operator.resolveGradients(gradient, operands);
		}

		public Value plus(Value that) {
			return calc(Operator.PLUS, that);
		}

		public Value plus(double value) {
			return plus(Value.of(value));
		}

		public Value minus(Value that) {
			return calc(Operator.MINUS, that);
		}

		public Value minus(double value) {
			return minus(Value.of(value));
		}

		public Value mult(Value that) {
			return calc(Operator.MULT, that);
		}

		public Value mult(double value) {
			return mult(Value.of(value));
		}

		public Value div(Value that) {
			return mult(that.pow(-1));
		}

		public Value div(double value) {
			return div(Value.of(value));
		}

		public Value pow(Value that) {
			return calc(Operator.POW, that);
		}

		public Value pow(double value) {
			return pow(Value.of(value));
		}

		public Value exp() {
			return calc(Operator.EXP);
		}

		public Value calc(Operator operator) {
			Double computed = operator.compute(Stream.of(this).map(Value::data).map(AtomicReference::get).toList());
			AtomicReference<Double> data = new AtomicReference<>(computed);
			return new Value("" + operator + "(" + this.data().get() + ")", data, operator, List.of(this),
					new AtomicReference<>());
		}

		public Value calc(Operator operator, Value that) {
			Double computed = operator
					.compute(Stream.of(this, that).map(Value::data).map(AtomicReference::get).toList());
			AtomicReference<Double> data = new AtomicReference<>(computed);
			return new Value("" + this.data().get() + " " + operator + " " + that.data().get(), data, operator,
					List.of(this, that), new AtomicReference<>());
		}

		@Override
		public final String toString() {
			return label;
		}

		public static Value of(double value) {
			return new Value(Double.toString(value), value);
		}

		public void backward() {
			resetGradientsRecursively();
			computeGradientsRecursivelyStartingHereAt(1.0);
		}
	}

	record Neuron(List<Value> weights, Value bias) {
		public Neuron(ParameterNamer parameterNamer, int inputSize, Function<String, Double> paramInitializer) {
			this(IntStream.range(0, inputSize).mapToObj(i -> new Value(parameterNamer.atX(i), paramInitializer))
					.toList(), new Value(parameterNamer.atBias(), paramInitializer));
		}

		public Value weight(int index) {
			return weights.get(index);
		}

		public Value compute(List<Value> x) {
			Value dotProduct = IntStream.range(0, x.size())//
					.mapToObj(i -> weights.get(i).mult(x.get(i)))//
					.reduce(Value.of(0), Value::plus);
			Value activity = dotProduct.plus(bias);
			return activity.calc(Operator.TANH);
		}

		public List<Value> parameters() {
			return Stream.concat(weights.stream(), Stream.of(bias)).toList();
		}
	}

	record Layer(List<Neuron> neurons) {
		public Layer(ParameterNamer parameterNamer, int inputSize, int neuronsCount,
				Function<String, Double> paramInitializer) {
			this(IntStream.range(0, neuronsCount)//
					.mapToObj(i -> new Neuron(parameterNamer.atNeuron(i), inputSize, paramInitializer))//
					.toList());
		}

		public Neuron neuron(int index) {
			return neurons.get(index);
		}

		public List<Value> compute(List<Value> x) {
			return neurons.stream().map(neuron -> neuron.compute(x)).toList();
		}

		public List<Value> parameters() {
			return neurons.stream().map(Neuron::parameters).flatMap(List::stream).toList();
		}
	}

	record MLP(List<Layer> layers) {
		private MLP(ParameterNamer parameterNamer, List<Integer> layersInputSizes, List<Integer> layersNeuronsCounts,
				Function<String, Double> paramInitializer) {
			this(IntStream.range(0, layersNeuronsCounts.size())//
					.mapToObj(i -> new Layer(parameterNamer.atLayer(i), layersInputSizes.get(i),
							layersNeuronsCounts.get(i), paramInitializer))//
					.toList());
		}

		public MLP(ParameterNamer parameterNamer, int inputSize, List<Integer> layersNeuronsCounts,
				Function<String, Double> paramInitializer) {
			this(parameterNamer,
					Stream.concat(Stream.of(inputSize),
							layersNeuronsCounts.subList(0, layersNeuronsCounts.size() - 1).stream()).toList(),
					layersNeuronsCounts, paramInitializer);
		}

		public Layer layer(int index) {
			return layers.get(index);
		}

		public List<Value> compute(List<Value> x) {
			return layers.stream().reduce(x, (vector, layer) -> layer.compute(vector), noCombiner());
		}

		public List<Double> computeRaw(List<Double> x) {
			return compute(x.stream().map(Value::of).toList()).stream().map(Value::data).map(AtomicReference::get)
					.toList();
		}

		public List<Value> parameters() {
			return layers.stream().map(Layer::parameters).flatMap(List::stream).toList();
		}

		public Value computeLoss(Map<List<Double>, Double> dataset) {
			Map<Value, Double> results = dataset.entrySet().stream().collect(toMap(//
					datapoint -> this.compute(datapoint.getKey().stream().map(Value::of).toList()).get(0), //
					datapoint -> datapoint.getValue()//
			));
			Value loss = Value.of(0);
			for (Entry<Value, Double> entry : results.entrySet()) {
				Value prediction = entry.getKey();
				Double target = entry.getValue();
				Value localLoss = prediction.minus(target).pow(2);
				loss = loss.plus(localLoss);
			}
			return loss;
		}

		public void updateParameters(double step) {
			for (Value parameter : this.parameters()) {
				parameter.data().accumulateAndGet(-step * parameter.gradient().get(), (a, b) -> a + b);
			}
		}

	}

	public static void main(String[] args) {
		Random random = new Random(0);
		// TODO Provide separate outputs (like inputs)
		// TODO Parallelize computation
		MLP mlp = switch (1) {
		case 1 -> new MLP(ParameterNamer.create(), 2, List.of(4, 1), (_) -> random.nextDouble(-1.0, 1.0));
		case 2 -> new MLP(ParameterNamer.create(), 2, List.of(20, 1), (_) -> random.nextDouble(-1.0, 1.0));
		case 3 -> new MLP(ParameterNamer.create(), 2, List.of(4, 4, 4, 4, 1), (_) -> random.nextDouble(-1.0, 1.0));
		default -> throw new IllegalArgumentException("Unexpected MLP");
		};
		Map<List<Double>, Double> dataset = switch (1) {
		case 1 -> Dataset.circle(random);
		case 2 -> Dataset.columns1(random);
		case 3 -> Dataset.columns2(random);
		case 4 -> Dataset.steepColumns(random);
		case 5 -> Dataset.moons(random);
		default -> throw new IllegalArgumentException("Unexpected dataset outputs");
		};

		AtomicReference<Double> updateStep = new AtomicReference<Double>(0.001);
		Supplier<RoundData> mlpRound = () -> {
			Instant start = Instant.now();
			Value loss = mlp.computeLoss(dataset);
			Instant computeTime = Instant.now();
			loss.backward();
			Instant backwardTime = Instant.now();
			mlp.updateParameters(updateStep.get());
			Instant updateTime = Instant.now();
			return new RoundData(//
					loss, //
					Duration.between(start, computeTime), //
					Duration.between(computeTime, backwardTime), //
					Duration.between(backwardTime, updateTime)//
			);
		};

		AtomicReference<Optional<Long>> roundsLimit = new AtomicReference<>(Optional.empty());
		AtomicLong batchSize = new AtomicLong(1);
		TrainConf trainConf = new TrainConf(roundsLimit, batchSize, updateStep);

		Collection<VisualDatasetConf> datasetConfs = List.of(//
				new VisualDatasetConf("1.0", value -> value > 0, Color.RED), //
				new VisualDatasetConf("-1.0", value -> value < 0, Color.BLUE)//
		);

		Resolution contourResolution = new Resolution(100, 100);
		BiFunction<Color, Double, Color> colorTransformation = (color, value) -> adaptSaturation(color,
				Math.abs(value) * 0.6);
		BinaryOperator<Double> contourFunction = (x, y) -> {
			return mlp.computeRaw(List.of(x, y)).get(0);
		};
		ContourConf contourConf = new ContourConf("MLP", contourResolution, colorTransformation, contourFunction);

		int xIndex = 0;
		int yIndex = 1;
		Color defaultColor = Color.BLACK;// transparent
		VisualConf visualConf = new VisualConf(xIndex, yIndex, defaultColor, datasetConfs, contourConf);

		PlotUtils.WindowFactory windowFactory = switch (2) {
		case 1 -> PlotUtils.createNoWindowFactory();
		case 2 -> PlotUtils.createFixedWindowFactory(100);
		case 3 -> PlotUtils.createSlidingWindowFactory(10);
		default -> throw new IllegalArgumentException("Unexpected window factory");
		};
		RectangleEdge legendPosition = RectangleEdge.TOP;
		TimePlotConf timePlotConf = new TimePlotConf(windowFactory, legendPosition);

		LossPlotConf lossPlotConf = new LossPlotConf(windowFactory);

		FrameConf frameConf = new FrameConf(trainConf, visualConf, lossPlotConf, timePlotConf);

		createFrame(frameConf, dataset, mlp, mlpRound);
	}

	private static Color adaptSaturation(Color color, double factor) {
		float[] hsb = Color.RGBtoHSB(color.getRed(), color.getGreen(), color.getBlue(), null);
		hsb[1] *= factor;
		return Color.getHSBColor(hsb[0], hsb[1], hsb[2]);
	}

	record Parts(JPanel panel, Consumer<List<RoundResult>> panelUpdater) {
	}

	private static void createFrame(FrameConf frameConf, Map<List<Double>, Double> dataset, MLP mlp,
			Supplier<RoundData> mlpRound) {
		Parts visualParts = createVisual(frameConf, dataset);

		Parts lossPlotParts = createLossPlot(frameConf.lossPlotConf());

		Parts timePlotParts = createTimePlot(frameConf.timePlotConf());

		JPanel mlpPanel;
		Consumer<List<RoundResult>> mlpUpdater;
		{
//			JLabel label = new JLabel();
//			JScrollPane pane = new JScrollPane(label);
			mlpPanel = new JPanel();
			mlpPanel.setLayout(new GridLayout(1, 1));
			JSVGCanvas svgCanvas = new JSVGCanvas();
			mlpPanel.add(svgCanvas);
//			mlpPanel.add(pane);
			// TODO Produce Loss SVG on demand
			// TODO Produce MLP SVG on demand
			// TODO Paint MLP only if shown
			{
				String fileName = "graph";
				Path dotPath = createTempPath(fileName, "dot");
				createMlpDot(mlp, dotPath);

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
			}
			// A deep copy might be done, so retrieve actual instance after copy
			SVGDocument svgDocument = svgCanvas.getSVGDocument();

			SVGSVGElement root = svgDocument.getRootElement();

			mlpUpdater = roundResults -> {
				if (true) {
					svgCanvas.getUpdateManager().getUpdateRunnableQueue().invokeLater(() -> {
						streamOf(root.getElementsByTagName("g"))//
						.filter(g -> titleOf(g).equals("I0->N0"))//
						.forEach(g -> {
							// TODO Retrieve weight from ID of neuron and ID of input
							Double weight = mlp.layer(0).neuron(0).weight(0).data().get();
							Node weightNode = findFirstChild(g, "text").orElseThrow();
							// TODO Use app locale?
							weightNode.setTextContent(String.format(Locale.US, "%.4f", weight));
							System.out.println(titleOf(g) + " = " + weightOf(g));
						});
					});
				} else {
					String fileName = "graph";
					Path dotPath = createTempPath(fileName, "dot");
					createCalculationDot(roundResults.getLast().data().loss(), dotPath);

					Path svgPath = createTempPath(fileName, "svg");
					dotToFile(dotPath, svgPath, "svg");
					System.out.println("Graph: " + svgPath);

					Path pngPath = createTempPath(fileName, "png");
					BufferedImage image = dotToImage(dotPath, pngPath);

					ImageIcon icon = new ImageIcon(image);
//						label.setIcon(icon);
				}
				;
			};
		}

		Consumer<List<RoundResult>> roundConsumer = mlpUpdater//
				.andThen(lossPlotParts.panelUpdater())//
				.andThen(timePlotParts.panelUpdater())//
				.andThen(visualParts.panelUpdater());
		JPanel trainPanel = createTrainPanel(frameConf, mlpRound, roundConsumer);

		JTabbedPane tabs = new JTabbedPane();
		tabs.add(visualParts.panel(), "Visual");
		tabs.add(lossPlotParts.panel(), "Loss plot");
		tabs.add(timePlotParts.panel(), "Time plot");
		tabs.add(mlpPanel, "MLP");
		tabs.add(new JPanel(), "empty");

		tabs.setSelectedComponent(mlpPanel);// TODO Remove

		JFrame frame = new JFrame("MLP");
		frame.setLayout(new GridBagLayout());
		{
			GridBagConstraints constraints = new GridBagConstraints();
			constraints.gridx = 0;
			constraints.gridy = GridBagConstraints.RELATIVE;
			constraints.weightx = 1.0;
			constraints.weighty = 1.0;
			constraints.fill = GridBagConstraints.BOTH;
			frame.add(tabs, constraints);
			constraints.weightx = 1.0;
			constraints.weighty = 0.0;
			constraints.fill = GridBagConstraints.HORIZONTAL;
			frame.add(trainPanel, constraints);
		}

		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.pack();
		frame.setSize(840, 930);
		frame.setVisible(true);
	}

	private static double weightOf(Node g) {
		return findFirstChild(g, "text").map(Node::getTextContent).map(Double::parseDouble).orElseThrow();
	}

	private static String titleOf(Node g) {
		return findFirstChild(g, "title").map(Node::getTextContent).orElse("∅");
	}

	private static Optional<Node> findFirstChild(Node g, String tag) {
		return streamOf(g.getChildNodes()).filter(child -> child.getNodeName().equals(tag)).findFirst();
	}

	private static Stream<Node> streamOf(NodeList nodeList) {
		return IntStream.range(0, nodeList.getLength()).mapToObj(nodeList::item);
	}

	private static Stream<Node> streamOf(NamedNodeMap nodeMap) {
		return IntStream.range(0, nodeMap.getLength()).mapToObj(nodeMap::item);
	}

	private static void createMlpDot(MLP mlp, Path dotPath) {
		record Input(int i) {
		}
		Function<Input, String> inputIdSupplier = memoize(prefixedCounterId("I"));
		Function<Neuron, String> neuronIdSupplier = memoize(prefixedCounterId("N"));
		Function<Layer, String> layerIdSupplier = memoize(prefixedCounterId("L"));
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
			dotWriter.println("color=lightgrey;");
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
				dotWriter.println("color=lightgrey;");
				dotWriter.println("label = \"" + layerId + "\";");
				layer.neurons().forEach(neuron -> {
					currentLayer.add(neuron);
					String neuronId = neuronIdSupplier.apply(neuron);
					ids.put(neuron, neuronId);
					dotWriter.println(neuronId + " [label=\"" + neuronId + "\", shape=circle];");
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

	private static <T> Function<T, String> prefixedCounterId(String prefix) {
		AtomicInteger counter = new AtomicInteger();
		return _ -> prefix + counter.getAndIncrement();
	}

	private static JPanel createTrainPanel(FrameConf frameConf, Supplier<RoundData> mlpRound,
			Consumer<List<RoundResult>> roundConsumer) {
		JTextField roundsLimitField = FieldBuilder.buildFieldFor(frameConf.trainConf().roundsLimit())//
				.intoText(src -> src.get().map(Object::toString).orElse(""))//
				.as(Long::parseLong).ifIs(value -> value > 0).thenApply((src, value) -> src.set(Optional.of(value)))//
				.whenEmptyApply(src -> src.set(Optional.empty()))//
				.otherwiseShow(FieldBuilder::error)//
				.build();

		JTextField batchSizeField = FieldBuilder.buildFieldFor(frameConf.trainConf().batchSize())//
				.intoText(src -> Long.toString(src.get()))//
				.as(Long::parseLong).ifIs(value -> value > 0).thenApply(AtomicLong::set)//
				.otherwiseShow(FieldBuilder::error)//
				.build();

		JTextField updateStepField = FieldBuilder.buildFieldFor(frameConf.trainConf().updateStep())//
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

		JToggleButton trainButton = new JToggleButton();
		roundConsumer = lossFieldUpdater.andThen(roundConsumer);
		trainButton.setAction(createTrainAction(frameConf, mlpRound, roundConsumer, trainButton));

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

	private static Parts createVisual(FrameConf frameConf, Map<List<Double>, Double> dataset) {
		VisualConf visualConf = frameConf.visualConf();
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

	private static AbstractAction createTrainAction(FrameConf frameConf, Supplier<RoundData> mlpRound,
			Consumer<List<RoundResult>> roundConsumer, JToggleButton runButton) {
		return new AbstractAction("Train") {
			long round = 0;

			@Override
			public void actionPerformed(ActionEvent e) {
				if (!runButton.isSelected()) {
					// Wait for the stuff to stop
				} else {
					AtomicReference<Optional<Long>> roundsLimit = frameConf.trainConf().roundsLimit();
					AtomicLong batchSize = frameConf.trainConf().batchSize();
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
							runButton.setSelected(false);
						}

						private boolean isTrainingStopped() {
							return !runButton.isSelected();
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

	record TrainConf(//
			AtomicReference<Optional<Long>> roundsLimit, //
			AtomicLong batchSize, //
			AtomicReference<Double> updateStep//
	) {
	}

	record VisualDatasetConf(//
			String label, //
			Predicate<Double> predicate, //
			Color color//
	) {
	}

	record VisualConf(//
			int xIndex, //
			int yIndex, //
			Color defaultColor, //
			Collection<VisualDatasetConf> datasetConfs, //
			ContourConf contourConf//
	) {
	}

	record ContourConf(//
			String name, //
			Resolution resolution, //
			BiFunction<Color, Double, Color> colorTransformation, //
			BinaryOperator<Double> function //
	) {
	}

	record TimePlotConf(PlotUtils.WindowFactory windowFactory, RectangleEdge legendPosition) {
	}

	record LossPlotConf(PlotUtils.WindowFactory windowFactory) {
	}

	record FrameConf(//
			TrainConf trainConf, //
			VisualConf visualConf, //
			LossPlotConf lossPlotConf, //
			TimePlotConf timePlotConf//
	) {
	}

	static double polynom(double x, Iterable<Double> factors) {
		Iterator<Double> iterator = factors.iterator();

		if (iterator.hasNext()) {
			double result = iterator.next();
			while (iterator.hasNext()) {
				result = result * x + iterator.next();
			}
			return result;
		} else {
			return 0;
		}
	}

	record Resolution(int x, int y) {
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

	private static ContourDimension createContourDimension(ValueAxis domainAxis, int resolutionX) {
		double minX = domainAxis.getLowerBound();
		double maxX = domainAxis.getUpperBound();
		return new ContourDimension(minX, maxX, resolutionX);
	}

	private static Path createTempPath(String fileName, String ext) {
		try {
			return Files.createTempFile(fileName, "." + ext);
		} catch (IOException cause) {
			throw new RuntimeException(cause);
		}
	}

	private static BufferedImage dotToImage(Path dotPath, Path pngPath) {
		dotToFile(dotPath, pngPath, "png");
		BufferedImage image;
		try {
			image = ImageIO.read(pngPath.toFile());
		} catch (IOException cause) {
			throw new RuntimeException(cause);
		}
		return image;
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

	private static void createCalculationDot(Value value, Path dotPath) {
		File dotFile = dotPath.toFile();
		AtomicInteger counter = new AtomicInteger();
		Function<Value, String> valueIdSupplier = memoize((_) -> "N" + counter.getAndIncrement());
		try (PrintWriter dotWriter = new PrintWriter(dotFile)) {
			dotWriter.println("digraph G {");
			dotWriter.println("rankdir=LR");// Left to right
			writeValueRecursively(value, dotWriter, valueIdSupplier, new HashSet<String>());
			dotWriter.println("}");
		} catch (FileNotFoundException cause) {
			throw new RuntimeException(cause);
		}
	}

	private static <T, U> Function<T, U> memoize(Function<T, U> function) {
		Map<T, U> cache = new HashMap<T, U>();
		return value -> cache.computeIfAbsent(value, function);
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
				value.operands.forEach(operand -> {
					String childId = writeValueRecursively(operand, dotWriter, valueIdSupplier, ids);
					dotWriter.println(childId + " -> " + operatorId + ";");
				});
			}
			ids.add(id);
		}
		return id;
	}

	private static String formatData(AtomicReference<Double> data) {
		return Optional.ofNullable(data.get()).map(NeuralNet::roundForDot).map(Object::toString).orElse("∅");
	}

	private static double roundForDot(double value) {
		return (double) Math.round(value * 10000) / 10000;
	}

	static enum X {
		INTEGRATED, VERTICAL
	}

	record SeriesDefinition(List<Double> xs, List<Double> ys, String lineTitle) {
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

	private static <T> void restrict(List<T> list, int expectedSize) {
		int actualSize = list.size();
		if (actualSize != expectedSize) {
			throw new IllegalArgumentException("Expect " + expectedSize + " items but got " + actualSize);
		}
	}

	private static <T> BinaryOperator<T> noCombiner() {
		return (_, _) -> {
			throw new RuntimeException("Not implemented");
		};
	}

	interface ParameterNamer extends Supplier<String> {
		String get();

		public static ParameterNamer create() {
			return new ParameterNamer() {
				@Override
				public String get() {
					return "";
				}
			};
		}

		default public ParameterNamer append(String postfix) {
			return () -> get() + postfix;
		}

		default public ParameterNamer atLayer(int i) {
			return append("L" + i);
		}

		default public ParameterNamer atNeuron(int i) {
			return append("N" + i);
		}

		default public ParameterNamer atX(int i) {
			return append("X" + i);
		}

		default public ParameterNamer atBias() {
			return append("B");
		}
	}

	record ContourDimension(double min, double max, int resolution) {
		public double step() {
			return (max() - min()) / resolution();
		}
	}

	record RoundData(Value loss, Duration computeDuration, Duration backwardDuration, Duration updateDuration) {
	}

	record RoundResult(long round, RoundData data) {
	}
}
