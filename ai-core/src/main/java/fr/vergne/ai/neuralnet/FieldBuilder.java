package fr.vergne.ai.neuralnet;

import java.awt.Color;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;

import javax.swing.JTextField;

interface FieldBuilder<T> {
	static <T> FieldBuilder<T> buildFieldFor(T source) {
		return new FieldBuilder<>() {
			@Override
			public TextFieldBuilder<T> toText(Function<T, String> toText) {
				return new TextFieldBuilder<>() {

					@Override
					public <U> SetterBuilder<T, U> whenUpdate(Function<String, U> textToValue) {
						return new SetterBuilder<>() {
							Predicate<U> valueChecker;

							@Override
							public SetterBuilder<T, U> andHas(Predicate<U> valueChecker) {
								this.valueChecker = valueChecker;
								return this;
							}

							@Override
							public TextFieldBuilder<T> thenSet(BiConsumer<T, U> valueConsumer) {
								return new TextFieldBuilder<>() {

									@Override
									public <U2> SetterBuilder<T, U2> whenUpdate(Function<String, U2> textToValue) {
										// TODO Decouple
										throw new RuntimeException("Decouple");
									}

									BiConsumer<JTextField, T> noTextDefault;

									@Override
									public TextFieldBuilder<T> whenEmptySet(Consumer<T> noTextDefault) {
										this.noTextDefault = (a, b) -> noTextDefault.accept(b);
										return this;
									}

									@Override
									public TextFieldBuilder<T> whenEmptyShow(Consumer<JTextField> noTextDefault) {
										this.noTextDefault = (a, b) -> noTextDefault.accept(a);
										return this;
									}

									Consumer<JTextField> noCheckDefault;

									@Override
									public TextFieldBuilder<T> otherwiseShow(Consumer<JTextField> noCheckDefault) {
										this.noCheckDefault = noCheckDefault;
										return this;
									}

									@Override
									public JTextField build() {
										JTextField textField = new JTextField(toText.apply(source));

										Runnable textUpdater = NeuralNet.createTextUpdater(//
												textField, //
												textToValue, //
												valueChecker, //
												value -> valueConsumer.accept(source, value), //
												() -> noTextDefault.accept(textField, source), //
												noCheckDefault//
										);

										NeuralNet.registerTextUpdater(textField, textUpdater);

										return textField;
									}
								};
							}
						};
					}

					@Override
					public TextFieldBuilder<T> whenEmptySet(Consumer<T> noTextDefault) {
						// TODO Decouple
						throw new RuntimeException("Decouple");
					}

					@Override
					public TextFieldBuilder<T> whenEmptyShow(Consumer<JTextField> noTextDefault) {
						// TODO Decouple
						throw new RuntimeException("Decouple");
					}

					@Override
					public TextFieldBuilder<T> otherwiseShow(Consumer<JTextField> noCheckDefault) {
						// TODO Decouple
						throw new RuntimeException("Decouple");
					}

					@Override
					public JTextField build() {
						// TODO Decouple
						throw new RuntimeException("Decouple");
					}
				};
			}

		};
	}

	TextFieldBuilder<T> toText(Function<T, String> toText);

	interface TextFieldBuilder<T> {
		<U> SetterBuilder<T, U> whenUpdate(Function<String, U> textToValue);

		interface SetterBuilder<T, U> {

			SetterBuilder<T, U> andHas(Predicate<U> valueChecker);

			TextFieldBuilder<T> thenSet(BiConsumer<T, U> valueConsumer);
		}

		TextFieldBuilder<T> whenEmptySet(Consumer<T> noTextDefault);

		TextFieldBuilder<T> whenEmptyShow(Consumer<JTextField> noTextDefault);

		TextFieldBuilder<T> otherwiseShow(Consumer<JTextField> noCheckDefault);

		JTextField build();
	}

	static Runnable error(JTextField textField) {
		return () -> {
			// Set text field background color to red to indicate error
			textField.setBackground(Color.RED);
		};
	}

}