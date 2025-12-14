package fr.vergne.ai.neuralnet;

import java.util.Arrays;
import java.util.function.BiConsumer;

interface PlotUtils {

	interface WindowFactory {
		PlotUtils.Window<Long, Long> createLongLongWindow(BiConsumer<Long, Long> consumer);

		PlotUtils.Window<Long, Double> createLongDoubleWindow(BiConsumer<Long, Double> consumer);
	}

	interface Window<T extends Number, U extends Number> {
		void feedWindow(T x, U y);
	}

	public static PlotUtils.WindowFactory createNoWindowFactory() {
		return new PlotUtils.WindowFactory() {

			@Override
			public PlotUtils.Window<Long, Long> createLongLongWindow(BiConsumer<Long, Long> consumer) {
				return new PlotUtils.Window<>() {
					@Override
					public void feedWindow(Long x, Long y) {
						consumer.accept(x, y);
					}
				};
			}

			@Override
			public Window<Long, Double> createLongDoubleWindow(BiConsumer<Long, Double> consumer) {
				return new PlotUtils.Window<>() {
					@Override
					public void feedWindow(Long x, Double y) {
						consumer.accept(x, y);
					}
				};
			}
		};
	}

	public static PlotUtils.WindowFactory createFixedWindowFactory(int windowSize) {
		return new PlotUtils.WindowFactory() {

			@Override
			public PlotUtils.Window<Long, Long> createLongLongWindow(BiConsumer<Long, Long> consumer) {
				return new PlotUtils.Window<>() {
					private final long[] window = new long[windowSize];

					@Override
					public void feedWindow(Long x, Long y) {
						int windowIndex = (int) (x % windowSize);
						window[windowIndex] = y;
						if (windowIndex == windowSize - 1) {
							long average = (long) Arrays.stream(window).average().getAsDouble();
							consumer.accept(x, average);
						} else {
							// Window not full yet
						}
					}
				};
			}

			@Override
			public Window<Long, Double> createLongDoubleWindow(BiConsumer<Long, Double> consumer) {
				return new PlotUtils.Window<>() {
					private final double[] window = new double[windowSize];

					@Override
					public void feedWindow(Long x, Double y) {
						int windowIndex = (int) (x % windowSize);
						window[windowIndex] = y;
						if (windowIndex == windowSize - 1) {
							double average = Arrays.stream(window).average().getAsDouble();
							consumer.accept(x, average);
						} else {
							// Window not full yet
						}
					}
				};
			}

		};
	}

	public static PlotUtils.WindowFactory createSlidingWindowFactory(int windowSize) {
		return new PlotUtils.WindowFactory() {

			@Override
			public PlotUtils.Window<Long, Long> createLongLongWindow(BiConsumer<Long, Long> consumer) {
				return new PlotUtils.Window<>() {
					private final long[] window = new long[windowSize];

					@Override
					public void feedWindow(Long x, Long y) {
						int windowIndex = (int) (x % windowSize);
						window[windowIndex] = y;
						long average = (long) Arrays.stream(window).average().getAsDouble();
						consumer.accept(x, average);
					}
				};
			}

			@Override
			public Window<Long, Double> createLongDoubleWindow(BiConsumer<Long, Double> consumer) {
				return new PlotUtils.Window<>() {
					private final double[] window = new double[windowSize];

					@Override
					public void feedWindow(Long x, Double y) {
						int windowIndex = (int) (x % windowSize);
						window[windowIndex] = y;
						double average = Arrays.stream(window).average().getAsDouble();
						consumer.accept(x, average);
					}
				};
			}
		};
	}
}