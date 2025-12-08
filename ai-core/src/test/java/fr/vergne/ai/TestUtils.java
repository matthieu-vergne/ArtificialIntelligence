package fr.vergne.ai;

import static org.mockito.Mockito.mock;

import java.util.function.Function;

import org.mockito.Mockito;

public class TestUtils {
	/**
	 * Helper method to name the provided {@link Function} based on a given name.
	 * 
	 * @param <T>      type of object to call
	 * @param <U>      type of returned value
	 * @param name     the name of the {@link Function}
	 * @param function the {@link Function} to name
	 * @return a new {@link Function} with an adapted {@link Object#toString()}
	 */
	public static <T, U> Function<T, U> name(String name, Function<T, U> function) {
		return new Function<T, U>() {
			@Override
			public U apply(T value) {
				return function.apply(value);
			}

			@Override
			public String toString() {
				return name;
			}
		};
	}

	/**
	 * Helper method to name the provided {@link Function} based on the method it
	 * calls. We return a new {@link Function} which runs the provided one, but
	 * override the {@link Object#toString()} method to return the name of the
	 * method called.
	 * <p>
	 * If a chain of calls happens, only the method name from the input object is
	 * retrieved. In case where the {@link Function} calls several methods of the
	 * input object, the name retrieved is from the first call. Beware that chained
	 * calls on the same instance are all considered, but a chain call that returns
	 * a new instance is ignored.
	 * 
	 * @param clazz    {@link Class} of T
	 * @param function the {@link Function} calling the method of the object
	 * 
	 * @param <T>      type of object to call
	 * @param <U>      type of returned value
	 * @return a new {@link Function} with an adapted {@link Object#toString()}
	 */
	public static <T, U> Function<T, U> name(Class<T> clazz, Function<T, U> function) {
		var ctx = new Object() {
			String methodName;
		};
		T value = mock(clazz, invocation -> {
			if (ctx.methodName == null) {
				ctx.methodName = invocation.getMethod().getName();
			}
			// Use deep stubs to avoid NPE for chained calls
			return Mockito.RETURNS_DEEP_STUBS.answer(invocation);
		});
		function.apply(value);
		return name(ctx.methodName, function);
	}
}
