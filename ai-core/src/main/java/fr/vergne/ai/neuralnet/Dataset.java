package fr.vergne.ai.neuralnet;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.BiFunction;

public interface Dataset {
	public static Map<List<Double>, Double> moons(Random random, int datasetSize, double noiseAmplitude) {
		BiFunction<Double, Double, Double> noise = (value, amplitude) -> {
			return value + random.nextDouble(-amplitude, amplitude);
		};
		Map<List<Double>, Double> dataset = new LinkedHashMap<>();
		int moonSize = datasetSize / 2;
		double step = 2.0 / moonSize;
		double x = -1.0;
		for (int i = 0; i < moonSize; i++) {
			x += step;
			double y1 = x * x - 0.7;
			double y2 = -y1;
			double dx = Math.PI / 10;
			dataset.put(List.of(noise.apply(x, noiseAmplitude) + dx, noise.apply(y1, noiseAmplitude)), 1.0);
			dataset.put(List.of(noise.apply(x, noiseAmplitude) - dx, noise.apply(y2, noiseAmplitude)), -1.0);
		}
		return dataset;
	}

	public static Map<List<Double>, Double> steepColumns(Random random, int datasetSize) {
		Map<List<Double>, Double> dataset = new LinkedHashMap<>();
		for (int i = 0; i < datasetSize; i++) {
			double x = random.nextDouble(-5, 5);
			double y = random.nextDouble(-5, 5);
			double value = Math.signum(y - Math.sin(10 * x) * 5);
			dataset.put(List.of(x, y), value);
		}
		return dataset;
	}

	public static Map<List<Double>, Double> columns2(Random random, int datasetSize) {
		Map<List<Double>, Double> dataset = new LinkedHashMap<>();
		for (int i = 0; i < datasetSize; i++) {
			double x = random.nextDouble(-5, 5);
			double y = random.nextDouble(-5, 5);
			double value = Math.signum(y - Math.sin(x) * 5);
			dataset.put(List.of(x, y), value);
		}
		return dataset;
	}

	public static Map<List<Double>, Double> columns1(Random random, int datasetSize) {
		Map<List<Double>, Double> dataset = new LinkedHashMap<>();
		for (int i = 0; i < datasetSize; i++) {
			double x = random.nextDouble(-5, 5);
			double y = random.nextDouble(-5, 5);
			double value = Math.signum(y - NeuralNet.polynom(x, List.of(-2.0, 5.0, 4.0, -3.0)));
			dataset.put(List.of(x, y), value);
		}
		return dataset;
	}

	public static Map<List<Double>, Double> circle(Random random, int datasetSize) {
		Map<List<Double>, Double> dataset = new LinkedHashMap<>();
		for (int i = 0; i < datasetSize; i++) {
			double x = random.nextDouble(-5, 5);
			double y = random.nextDouble(-5, 5);
			double value = Math.signum(Math.sqrt(x * x + y * y) - 3);
			dataset.put(List.of(x, y), value);
		}
		return dataset;
	}
}