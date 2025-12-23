package fr.vergne.ai.utils;

import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

public class LambdaUtils {

	public static <T, U> Function<T, U> memoize(Function<T, U> function) {
		Map<T, U> cache = new HashMap<T, U>();
		return value -> cache.computeIfAbsent(value, function);
	}

}
