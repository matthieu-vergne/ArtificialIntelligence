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
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
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
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTextField;
import javax.swing.JToggleButton;
import javax.swing.SwingUtilities;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.LookupPaintScale;
import org.jfree.chart.renderer.xy.XYBlockRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.DomainOrder;
import org.jfree.data.general.DatasetChangeListener;
import org.jfree.data.general.DatasetGroup;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.data.xy.XYZDataset;

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
		if (true) {
			Random random = new Random(0);
			// TODO Provide separate outputs (like inputs)
			// TODO Parallelize computation
			MLP mlp = new MLP(ParameterNamer.create(), 2, List.of(4, 1), (_) -> random.nextDouble(-1.0, 1.0));
//			MLP mlp = new MLP(ParameterNamer.create(), 2, List.of(20, 1), (_) -> random.nextDouble(-1.0, 1.0));
//			MLP mlp = new MLP(ParameterNamer.create(), 2, List.of(4, 4, 4, 4, 1), (_) -> random.nextDouble(-1.0, 1.0));
			Map<List<Double>, Double> dataset = new LinkedHashMap<>();
			for (int i = 0; i < 100; i++) {
				double x = random.nextDouble(-5, 5);
				double y = random.nextDouble(-5, 5);
				double value = Math.signum(Math.sqrt(x * x + y * y) - 3);
//				double value = Math.signum(y - polynom(x, List.of(-2.0, 5.0, 4.0, -3.0)));
//				double value = Math.signum(y - Math.sin(x) * 5);
//				double value = Math.signum(y - Math.sin(10 * x) * 5);
				dataset.put(List.of(x, y), value);
			}

			AtomicReference<Double> updateStep = new AtomicReference<Double>(0.01);
			Supplier<Value> mlpRound = () -> {
				Value loss = mlp.computeLoss(dataset);
				loss.backward();
				mlp.updateParameters(updateStep.get());
				return loss;
			};

			AtomicReference<Optional<Integer>> roundsLimit = new AtomicReference<>(Optional.empty());
			AtomicInteger batchSize = new AtomicInteger(1);
			RunConf runConf = new RunConf(roundsLimit, batchSize, updateStep);

			Collection<ChartDatasetConf> datasetConfs = List.of(//
					new ChartDatasetConf("1.0", value -> value > 0, Color.RED), //
					new ChartDatasetConf("-1.0", value -> value < 0, Color.BLUE)//
			);

			Resolution contourResolution = new Resolution(100, 100);
			BinaryOperator<Double> contourFunction = (x, y) -> {
				return mlp.computeRaw(List.of(x, y)).get(0);
			};
			ContourConf contourConf = new ContourConf("MLP", contourResolution, contourFunction);

			int xIndex = 0;
			int yIndex = 1;
			Color defaultColor = Color.WHITE;
			ChartConf chartConf = new ChartConf(xIndex, yIndex, defaultColor, datasetConfs, contourConf);

			FrameConf frameConf = new FrameConf(runConf, chartConf);

			createFrame(frameConf, dataset, mlp, mlpRound);
		} else {
			List<Double> xs = range(-5, 5, 0.25).toList();
			Function<Double, Double> f = Math::tanh;
			Function<Double, Double> g = derivativeOf(f);
			SeriesDefinition sf = new SeriesDefinition(xs, xs.stream().map(f).toList(), "tanh");
			SeriesDefinition sg = new SeriesDefinition(xs, xs.stream().map(g).toList(), "tanh'");
			plot(List.of(sf, sg), "x", "y", "tanh", "tanh", X.INTEGRATED);
		}
	}

	private static void createFrame(FrameConf frameConf, Map<List<Double>, Double> dataset, MLP mlp,
			Supplier<Value> mlpRound) {
		ChartPanel chartPanel = createChartPanel(frameConf.chartConf(), dataset);

		// TODO Plot loss over rounds

		JLabel label = new JLabel();
		JScrollPane pane = new JScrollPane(label);
		JPanel graphPanel = new JPanel();
		graphPanel.add(pane);
		Consumer<Value> lossConsumer = loss -> {
			// TODO Show MLP graphically
			// TODO Produce Loss SVG on demand
			// TODO Produce MLP SVG on demand
			// TODO Paint MLP only if shown
			if (false) {
				String fileName = "graph";
				Path dotPath = createTempPath(fileName, "dot");
				createDot(loss, dotPath);

				Path svgPath = createTempPath(fileName, "svg");
				dotToFile(dotPath, svgPath, "svg");
				System.out.println("Graph: " + svgPath);

				Path pngPath = createTempPath(fileName, "png");
				BufferedImage image = dotToImage(dotPath, pngPath);

				ImageIcon icon = new ImageIcon(image);
				label.setIcon(icon);
			}
		};

		JTextField roundsLimitField = FieldBuilder.buildFieldFor(frameConf.runConf().roundsLimit())//
				.intoText(src -> src.get().map(Object::toString).orElse(""))//
				.as(Integer::parseInt).ifIs(value -> value > 0).thenApply((src, value) -> src.set(Optional.of(value)))//
				.whenEmptyApply(src -> src.set(Optional.empty()))//
				.otherwiseShow(FieldBuilder::error)//
				.build();

		JTextField batchSizeField = FieldBuilder.buildFieldFor(frameConf.runConf().batchSize())//
				.intoText(src -> Integer.toString(src.get()))//
				.as(Integer::parseInt).ifIs(value -> value > 0).thenApply(AtomicInteger::set)//
				.otherwiseShow(FieldBuilder::error)//
				.build();

		JTextField updateStepField = FieldBuilder.buildFieldFor(frameConf.runConf().updateStep())//
				.intoText(src -> Double.toString(src.get()))//
				.as(Double::parseDouble).ifIs(value -> value > 0).thenApply(AtomicReference::set)//
				.otherwiseShow(FieldBuilder::error)//
				.build();

		JTextField roundField = new JTextField();
		JTextField lossField = new JTextField();
		BiConsumer<Long, Value> lossFieldUpdater = (round, loss) -> {
			roundField.setText(round.toString());
			lossField.setText(loss.data().get().toString());
		};

		JToggleButton runButton = new JToggleButton();
		runButton.setAction(createTrainAction(frameConf, mlpRound, chartPanel, lossFieldUpdater, runButton));

		JPanel runPanel = new JPanel();
		runPanel.setLayout(new GridBagLayout());
		{
			GridBagConstraints constraints = new GridBagConstraints();
			constraints.gridx = GridBagConstraints.RELATIVE;
			constraints.gridy = 0;
			constraints.insets = new Insets(5, 5, 5, 5);
			constraints.weighty = 0.0;
			constraints.fill = GridBagConstraints.HORIZONTAL;
			constraints.weightx = 1.0;
			runPanel.add(runButton, constraints);
			constraints.weightx = 1.0;
			runPanel.add(roundsLimitField, constraints);
			constraints.weightx = 0.0;
			runPanel.add(new JLabel("rounds by batch of"), constraints);
			constraints.weightx = 1.0;
			runPanel.add(batchSizeField, constraints);
			constraints.weightx = 0.0;
			runPanel.add(new JLabel("and step of"), constraints);
			constraints.weightx = 1.0;
			runPanel.add(updateStepField, constraints);
			constraints.weightx = 0.0;
			runPanel.add(new JLabel("|"), constraints);
			constraints.weightx = 0.0;
			runPanel.add(new JLabel("Round:"), constraints);
			constraints.weightx = 1.0;
			runPanel.add(roundField, constraints);
			constraints.weightx = 0.0;
			runPanel.add(new JLabel("Loss:"), constraints);
			constraints.weightx = 1.0;
			runPanel.add(lossField, constraints);
		}

		JTabbedPane screenPane = new JTabbedPane();
		screenPane.add(chartPanel, "chart");
		screenPane.add(graphPanel, "graph");
		screenPane.add(new JPanel(), "empty");
		// To show the headers with each card title

		JFrame frame = new JFrame("MLP");
		frame.setLayout(new GridBagLayout());
		{
			GridBagConstraints constraints = new GridBagConstraints();
			constraints.gridx = 0;
			constraints.gridy = GridBagConstraints.RELATIVE;
			constraints.weightx = 1.0;
			constraints.weighty = 1.0;
			constraints.fill = GridBagConstraints.BOTH;
			frame.add(screenPane, constraints);
			constraints.weightx = 1.0;
			constraints.weighty = 0.0;
			constraints.fill = GridBagConstraints.HORIZONTAL;
			frame.add(runPanel, constraints);
		}

		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.pack();
		frame.setSize(800, 600);
		frame.setVisible(true);
	}

	private static AbstractAction createTrainAction(FrameConf frameConf, Supplier<Value> mlpRound,
			ChartPanel chartPanel, BiConsumer<Long, Value> roundConsumer, JToggleButton runButton) {
		return new AbstractAction("Train") {
			long round = 0;

			@Override
			public void actionPerformed(ActionEvent e) {
				if (!runButton.isSelected()) {
					// Wait for the stuff to stop
				} else {
					Optional<Integer> roundsLimit = frameConf.runConf().roundsLimit().get();
					if (roundsLimit.isEmpty()) {
						SwingUtilities.invokeLater(new Runnable() {
							@Override
							public void run() {
								if (!runButton.isSelected()) {
									// Don't request any more run
								} else {
									int batchSize = frameConf.runConf().batchSize().get();
									for (int i = 0; i < batchSize && runButton.isSelected(); i++) {
										round++;
										Value loss = mlpRound.get();
										roundConsumer.accept(round, loss);
									}
									chartPanel.getChart().fireChartChanged();
									chartPanel.repaint();
									SwingUtilities.invokeLater(this);
								}
							}
						});
					} else {
						// Snapshot the current number of rounds to not let it evolve during the run
						// We have the toggle to stop the run explicitly
						var ctx = new Object() {
							int rounds = roundsLimit.get();
						};
						SwingUtilities.invokeLater(new Runnable() {
							@Override
							public void run() {
								if (!runButton.isSelected()) {
									// Don't request any more run
								} else if (ctx.rounds == 0) {
									// Untoggle automatically
									runButton.setSelected(false);
								} else {
									int batchSize = Math.min(ctx.rounds, frameConf.runConf().batchSize().get());
									for (int i = 0; i < batchSize && runButton.isSelected(); i++) {
										round++;
										Value loss = mlpRound.get();
										roundConsumer.accept(round, loss);
										ctx.rounds--;
									}
									chartPanel.getChart().fireChartChanged();
									chartPanel.repaint();
									SwingUtilities.invokeLater(this);
								}
							}
						});
					}
				}
			}
		};
	}

	private static ChartPanel createChartPanel(ChartConf chartConf, Map<List<Double>, Double> dataset) {
		BiFunction<ChartDatasetConf, Integer, List<Double>> valuesExtractor = (datasetConf, index) -> {
			return dataset.entrySet().stream()//
					.filter(entry -> datasetConf.predicate().test(entry.getValue()))//
					.map(Entry::getKey)//
					.map(list -> list.get(index))//
					.toList();
		};
		Function<ChartDatasetConf, SeriesDefinition> seriesFactory = datasetConf -> {
			return new SeriesDefinition(//
					valuesExtractor.apply(datasetConf, chartConf.xIndex()), //
					valuesExtractor.apply(datasetConf, chartConf.yIndex()), //
					datasetConf.label());
		};
		Map<XYSeries, Color> coloredSeries = chartConf.datasetConfs().stream()//
				.collect(toMap(//
						datasetConf -> createSeries(seriesFactory.apply(datasetConf)), //
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

		addContour(chartConf.contourConf(), plot, createContourPainter(chartConf));

		return new ChartPanel(chart);
	}

	private static void addContour(ContourConf contourConf, XYPlot plot, Function<Double, Paint> contourPainter) {
		XYBlockRenderer contourRenderer = createContourRenderer(contourPainter);

		XYZDataset contourDataset = createContourDataset(contourConf, plot);

		int newIndex = plot.getDatasetCount();
		plot.setDataset(newIndex, contourDataset);
		plot.setRenderer(newIndex, contourRenderer);
	}

	private static Function<Double, Paint> createContourPainter(ChartConf chartConf) {
		Function<Double, Paint> contourPainter = value -> {
			Color datasetColor = chartConf.datasetConfs().stream()//
					.filter(datasetConf -> datasetConf.predicate().test(value))//
					.findAny()//
					.map(ChartDatasetConf::color).orElse(chartConf.defaultColor());
			return transparent(datasetColor);
		};
		return contourPainter;
	}

	private static XYBlockRenderer createContourRenderer(Function<Double, Paint> contourPainter) {
		XYBlockRenderer contourRenderer = new XYBlockRenderer();
		contourRenderer.setPaintScale(new LookupPaintScale() {
			@Override
			public Paint getPaint(double value) {
				return contourPainter.apply(value);
			}
		});
		return contourRenderer;
	}

	record RunConf(//
			AtomicReference<Optional<Integer>> roundsLimit, //
			AtomicInteger batchSize, //
			AtomicReference<Double> updateStep//
	) {
	}

	record ChartDatasetConf(//
			String label, //
			Predicate<Double> predicate, //
			Color color//
	) {
	}

	record ChartConf(//
			int xIndex, //
			int yIndex, //
			Color defaultColor, //
			Collection<ChartDatasetConf> datasetConfs, //
			ContourConf contourConf//
	) {
	}

	record ContourConf(//
			String name, //
			Resolution resolution, //
			BinaryOperator<Double> function //
	) {
	}

	record FrameConf(//
			RunConf runConf, //
			ChartConf chartConf //
	) {
	}

	private static double polynom(double x, Iterable<Double> factors) {
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

	private static Color transparent(Color red) {
		return new Color(red.getRed(), red.getGreen(), red.getBlue(), 1);
	}

	record Resolution(int x, int y) {
	}

	private static XYZDataset createContourDataset(ContourConf contourConf, XYPlot plot) {
		ContourDimension dimX = createContourDimension(plot.getDomainAxis(), contourConf.resolution().x());
		ContourDimension dimY = createContourDimension(plot.getRangeAxis(), contourConf.resolution().y());
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
				double x = xUnit * (dimX.max() - dimX.min()) + dimX.min();
				return x;
			}

			@Override
			public double getYValue(int series, int item) {
				int yItem = (item / dimX.resolution()) % dimY.resolution();
				double yUnit = (double) yItem / dimX.resolution();
				double y = yUnit * (dimY.max() - dimY.min()) + dimY.min();
				return y;
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
			public Comparable getSeriesKey(int series) {
				return contourConf.name();
			}

			@Override
			public int indexOf(Comparable seriesKey) {
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

	private static void createDot(Value value, Path dotPath) {
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

	private static double roundForDot(Double value) {
		return (double) Math.round(value * 10000) / 10000;
	}

	private static Function<Double, Double> derivativeOf(Function<Double, Double> f) {
		double epsilon = 1e-10;
		return x -> (f.apply(x + epsilon) - f.apply(x)) / epsilon;
	}

	static enum X {
		INTEGRATED, VERTICAL
	}

	private static void plot(List<SeriesDefinition> seriesDefinitions, String xTitle, String yTitle, String chartTitle,
			String windowTitle, X x) {
		switch (x) {
		case INTEGRATED -> {
			XYSeriesCollection dataset = new XYSeriesCollection();
			seriesDefinitions.stream().map(NeuralNet::createSeries).forEach(dataset::addSeries);
			JFreeChart chart = ChartFactory.createXYLineChart(chartTitle, // Title
					xTitle, // X-axis label
					yTitle, // Y-axis label
					dataset, // Dataset
					PlotOrientation.VERTICAL, // Orientation
					true, // Include legend
					true, // Tooltips
					false // URLs
			);

			// Display the chart in a frame
			ChartFrame frame = new ChartFrame(windowTitle, chart);
			frame.pack();
			frame.setVisible(true);
		}
		case VERTICAL -> {
			JFrame frame = new JFrame(windowTitle);
			frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
			frame.setLayout(new GridLayout(2, 1));
			seriesDefinitions.forEach(def -> {
				XYSeries series = createSeries(def);
				XYSeriesCollection dataset = new XYSeriesCollection();
				dataset.addSeries(series);
				JFreeChart chart = ChartFactory.createXYLineChart(chartTitle, // Title
						xTitle, // X-axis label
						yTitle, // Y-axis label
						dataset, // Dataset
						PlotOrientation.VERTICAL, // Orientation
						true, // Include legend
						true, // Tooltips
						false // URLs
				);
				frame.add(new ChartPanel(chart));
//				frame.add(new ChartFrame(windowTitle, chart));
			});
			frame.pack();
			frame.setVisible(true);
		}
		}
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

	private static Stream<Double> range(double min, double max, double increment) {
		int size = (int) ((max - min) / increment);
		return IntStream.range(0, size).mapToObj(i -> (double) i * increment + min);
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
	}
}
