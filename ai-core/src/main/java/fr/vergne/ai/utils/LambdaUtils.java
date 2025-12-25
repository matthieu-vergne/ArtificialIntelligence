package fr.vergne.ai.utils;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;

public class LambdaUtils {

	public static <T, U> Function<T, U> memoize(Function<T, U> function) {
		Map<T, U> cache = new HashMap<T, U>();
		return value -> cache.computeIfAbsent(value, function);
	}

	public static <T> Supplier<T> memoize(Supplier<T> supplier) {
		return new Supplier<T>() {
			T cache = null;

			@Override
			public T get() {
				if (cache == null) {
					cache = supplier.get();
				}
				return cache;
			}
		};
	}

}
