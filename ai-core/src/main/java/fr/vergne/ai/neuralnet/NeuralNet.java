package fr.vergne.ai.neuralnet;

import static java.util.stream.Collectors.toMap;

import java.awt.Color;
import java.time.Duration;
import java.time.Instant;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import javax.swing.JFrame;

import org.jfree.chart.ui.RectangleEdge;

import fr.vergne.ai.swing.App;
import fr.vergne.ai.swing.App.Conf;
import fr.vergne.ai.swing.App.ContourConf;
import fr.vergne.ai.swing.App.LossPlotConf;
import fr.vergne.ai.swing.App.NeuralNetConf;
import fr.vergne.ai.swing.App.Resolution;
import fr.vergne.ai.swing.App.RoundData;
import fr.vergne.ai.swing.App.TimePlotConf;
import fr.vergne.ai.swing.App.TrainConf;
import fr.vergne.ai.swing.App.VisualConf;
import fr.vergne.ai.swing.App.VisualDatasetConf;
import fr.vergne.ai.swing.PlotUtils;

public class NeuralNet {

	public enum Operator {
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

	public record Input(int i) {
	}

	public record Neuron(List<Value> weights, Value bias) {
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

	public record Layer(List<Neuron> neurons) {
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

	public record MLP(List<Layer> layers) {
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

		int displayedDecimals = 4;
		Color clusterColor = Color.LIGHT_GRAY;
		NeuralNetConf neuralNetConf = new NeuralNetConf(displayedDecimals, clusterColor);

		Conf conf = new Conf(trainConf, neuralNetConf, visualConf, lossPlotConf, timePlotConf);

		App app = new App(conf, dataset, mlp, mlpRound);
		app.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		app.pack();
		app.setSize(840, 930);
		app.setVisible(true);
	}

	private static Color adaptSaturation(Color color, double factor) {
		float[] hsb = Color.RGBtoHSB(color.getRed(), color.getGreen(), color.getBlue(), null);
		hsb[1] *= factor;
		return Color.getHSBColor(hsb[0], hsb[1], hsb[2]);
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

	public interface NeuronDescriptor {
		double weightWith(String inputId);

		double bias();
	}

	public interface NeuralNetBrowser {
		NeuronDescriptor neuron(String neuronId);

		public static NeuralNetBrowser forMlp(MLP mlp) {
			return new NeuralNetBrowser() {

				@Override
				public NeuronDescriptor neuron(String neuronId) {
					if (!neuronId.matches("N[0-9]+")) {
						throw new IllegalArgumentException("Unrecognized neuron ID: " + neuronId);
					}
					int neuronIndex = Integer.parseInt(neuronId.substring(1));
					Neuron neuron = retrieveNeuron(neuronIndex);
					if (neuron == null) {
						throw new IllegalArgumentException("Neuron ID too large: " + neuronId);
					}
					Layer outLayer = retrieveLayer(neuronIndex);
					return new NeuronDescriptor() {

						@Override
						public double bias() {
							return neuron.bias().data().get();
						}

						@Override
						public double weightWith(String inputId) {
							if (inputId.matches("I[0-9]+") && !outLayer.equals(mlp.layer(0))) {
								throw new IllegalArgumentException(
										"Only neuron of first hidden layer can use input: " + inputId);
							} else if (!inputId.matches("[IN][0-9]+")) {
								throw new IllegalArgumentException("Unrecognized neuron ID: " + inputId);
							}
							String inputType = inputId.substring(0, 1);
							if (inputType.equals("I")) {
								int inputIndex = Integer.parseInt(inputId.substring(1));
								Double inputWeight = retrieveInputWeight(neuron, inputIndex);
								if (inputWeight == null) {
									throw new IllegalArgumentException("Input ID too large: " + neuronId);
								}
								return inputWeight;
							} else if (inputType.equals("N")) {
								int inputIndex = Integer.parseInt(inputId.substring(1));
								Layer inLayer = retrieveLayer(inputIndex);
								if (mlp.layers().indexOf(inLayer) != mlp.layers().indexOf(outLayer) - 1) {
									throw new IllegalArgumentException(inputId + " is not an input of " + neuronId);
								}
								Double weight = retrieveNeuronWeight(neuron, inputIndex);
								if (weight == null) {
									throw new IllegalArgumentException("Input ID too large: " + neuronId);
								}
								return weight;
							} else {
								throw new RuntimeException("Unsupported input type: " + inputType);
							}
						}

						private Double retrieveNeuronWeight(Neuron neuron, int inputIndex) {
							for (Layer layer : mlp.layers()) {
								List<Neuron> neurons = layer.neurons();
								int neuronsCount = neurons.size();
								if (inputIndex >= neuronsCount) {
									inputIndex -= neuronsCount;
								} else {
									return neuron.weight(inputIndex).data().get();
								}
							}
							return null;
						}
					};
				}

				private Double retrieveInputWeight(Neuron neuron, int inputIndex) {
					Value value = neuron.weight(inputIndex);
					if (value == null) {
						return null;
					} else {
						return value.data().get();
					}
				}

				private Neuron retrieveNeuron(int neuronIndex) {
					for (Layer layer : mlp.layers()) {
						List<Neuron> neurons = layer.neurons();
						int neuronsCount = neurons.size();
						if (neuronIndex >= neuronsCount) {
							neuronIndex -= neuronsCount;
						} else {
							return layer.neuron(neuronIndex);
						}
					}
					return null;
				}

				private Layer retrieveLayer(int neuronIndex) {
					for (Layer layer : mlp.layers()) {
						List<Neuron> neurons = layer.neurons();
						int neuronsCount = neurons.size();
						if (neuronIndex >= neuronsCount) {
							neuronIndex -= neuronsCount;
						} else {
							return layer;
						}
					}
					return null;
				}
			};
		}
	}
}
