package fr.vergne.ai.neuralnet;

import java.awt.Color;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;

import javax.swing.JTextField;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;

interface FieldBuilder<T> {
	static <T> FieldBuilder<T> buildFieldFor(T source) {
		return new FieldBuilder<>() {
			@Override
			public TextFieldBuilder<T> intoText(Function<T, String> toText) {
				return new TextFieldBuilder<>() {

					private Consumer<JTextField> overallDefault;
					private BiConsumer<JTextField, T> noTextDefault = (textField, value) -> overallDefault
							.accept(textField);
					private BiConsumer<JTextField, String> textConsumer = (textField, text) -> overallDefault
							.accept(textField);

					@Override
					public <U> SetterBuilder<T, U> as(Function<String, U> textToValue) {
						var builder = this;
						return new SetterBuilder<>() {
							private Predicate<U> valueChecker;

							@Override
							public SetterBuilder<T, U> ifIs(Predicate<U> valueChecker) {
								this.valueChecker = valueChecker;
								return this;
							}

							@Override
							public TextFieldBuilder<T> thenApply(BiConsumer<T, U> valueConsumer) {
								builder.textConsumer = (textField, text) -> {
									U value = textToValue.apply(text);
									if (valueChecker.test(value)) {
										valueConsumer.accept(source, value);
									} else {
										overallDefault.accept(textField);
									}
								};
								return builder;
							}
						};
					}

					@Override
					public TextFieldBuilder<T> whenEmptyApply(Consumer<T> noTextDefault) {
						this.noTextDefault = (textField, value) -> noTextDefault.accept(value);
						return this;
					}

					@Override
					public TextFieldBuilder<T> whenEmptyShow(Consumer<JTextField> noTextDefault) {
						this.noTextDefault = (textField, value) -> noTextDefault.accept(textField);
						return this;
					}

					@Override
					public TextFieldBuilder<T> otherwiseShow(Consumer<JTextField> noCheckDefault) {
						this.overallDefault = noCheckDefault;
						return this;
					}

					@Override
					public JTextField build() {
						JTextField textField = new JTextField(toText.apply(source));
						Color defaultBackground = textField.getBackground();

						Runnable textUpdater = () -> {
							textField.setBackground(defaultBackground);
							String text = textField.getText();
							if (text.isEmpty()) {
								noTextDefault.accept(textField, source);
							} else {
								textConsumer.accept(textField, text);
							}
						};

						textField.getDocument().addDocumentListener(new DocumentListener() {

							@Override
							public void removeUpdate(DocumentEvent e) {
								textUpdater.run();
							}

							@Override
							public void insertUpdate(DocumentEvent e) {
								textUpdater.run();
							}

							@Override
							public void changedUpdate(DocumentEvent e) {
								textUpdater.run();
							}
						});

						return textField;
					}
				};
			}

		};
	}

	TextFieldBuilder<T> intoText(Function<T, String> toText);

	interface TextFieldBuilder<T> {
		<U> SetterBuilder<T, U> as(Function<String, U> textToValue);

		interface SetterBuilder<T, U> {

			SetterBuilder<T, U> ifIs(Predicate<U> valueChecker);

			TextFieldBuilder<T> thenApply(BiConsumer<T, U> valueConsumer);
		}

		TextFieldBuilder<T> whenEmptyApply(Consumer<T> noTextDefault);

		TextFieldBuilder<T> whenEmptyShow(Consumer<JTextField> noTextDefault);

		TextFieldBuilder<T> otherwiseShow(Consumer<JTextField> noCheckDefault);

		JTextField build();
	}

	static void error(JTextField textField) {
		// Set text field background color to red to indicate error
		textField.setBackground(Color.RED);
	}

}