package fr.vergne.ai.neuralnet;

import static java.util.stream.Collectors.toMap;

import java.awt.GridLayout;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
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
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JScrollPane;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class NeuralNet {

	enum Operator {
		NONE(null, //
				(operands) -> {
					throw new RuntimeException("Should not be called");
				}, //
				(parentGradient, operands) -> {
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

		public List<Value> computeRaw(List<Double> x) {
			return compute(x.stream().map(Value::of).toList());
		}

		public List<Value> parameters() {
			return layers.stream().map(Layer::parameters).flatMap(List::stream).toList();
		}

		public Value computeLoss(Map<List<Double>, Double> dataset) {
			Map<Value, Double> results = dataset.entrySet().stream().collect(toMap(//
					datapoint -> this.computeRaw(datapoint.getKey()).get(0), //
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
			MLP mlp = new MLP(ParameterNamer.create(), 3, List.of(4, 4, 1), (label) -> random.nextDouble(-1.0, 1.0));
			Map<List<Double>, Double> dataset = new LinkedHashMap<>();
			dataset.put(List.of(2.0, 3.0, -1.0), 1.0);
			dataset.put(List.of(3.0, -1.0, 0.5), -1.0);
			dataset.put(List.of(0.5, 1.0, 1.0), -1.0);
			dataset.put(List.of(1.0, 1.0, -1.0), 1.0);

			Value loss = null;
			for (int i = 0; i < 20; i++) {
				loss = mlp.computeLoss(dataset);
				System.out.println("Loss = " + loss.data().get());
				loss.backward();
				mlp.updateParameters(0.5);
			}

			if (loss != null) {
				graph(loss);
			}
		} else {
			List<Double> xs = range(-5, 5, 0.25).toList();
			Function<Double, Double> f = Math::tanh;
			Function<Double, Double> g = derivativeOf(f);
			SeriesDefinition sf = new SeriesDefinition(xs, xs.stream().map(f).toList(), "tanh");
			SeriesDefinition sg = new SeriesDefinition(xs, xs.stream().map(g).toList(), "tanh'");
			plot(List.of(sf, sg), "x", "y", "tanh", "tanh", X.INTEGRATED);
		}
	}

	private static void graph(Value L) {
		String fileName = "graph";
		Path dotPath = createTempPath(fileName, "dot");
		createDot(L, dotPath);

		Path svgPath = createTempPath(fileName, "svg");
		dotToFile(dotPath, svgPath, "svg");
		System.out.println("Graph: " + svgPath);

		Path pngPath = createTempPath(fileName, "png");
		BufferedImage image = dotToImage(dotPath, pngPath);

		JFrame frame = new JFrame("Graph");
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frame.setLayout(new GridLayout(1, 1));
		frame.add(new JScrollPane(new JLabel(new ImageIcon(image))));
		frame.pack();
		frame.setSize(800, 600);
		frame.setVisible(true);
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
		Function<Value, String> valueIdSupplier = memoize(val -> "N" + counter.getAndIncrement());
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
		return (a, b) -> {
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
}
