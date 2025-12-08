package fr.vergne.ai.ttt;

import java.util.NoSuchElementException;
import java.util.PrimitiveIterator.OfInt;

import fr.vergne.ai.ttt.TicTacToeBoard.Player;

public class TicTacToeFactory {

	public TicTacToeBoard createBoard(String string) {
		string = (string.toUpperCase() + "\n").replaceAll("\n\n$", "\n");
		// int rows = string.replaceAll("[^\n]", "").length();
		// int cols = string.replaceAll("^(.*)(?s:.*)$", "$1").length();

		TicTacToeBoard board = new TicTacToeBoard();
		board.reset();

		OfInt chars = string.chars().iterator();
		try {
			for (int row = 0; row < 3; row++) {
				for (int col = 0; col < 3; col++) {
					char character = (char) chars.nextInt();
					Player state;
					if (character == 'X') {
						state = Player.X;
					} else if (character == 'O') {
						state = Player.O;
					} else if (character == ' ') {
						state = null;
					} else {
						throw new InvalidBoardDescriptionException(string);
					}
					board.set(row, col, state);
				}
				char character = (char) chars.nextInt();
				if (character == '\n') {
					continue;
				} else {
					throw new InvalidBoardDescriptionException(string);
				}
			}
		} catch (NoSuchElementException e) {
			throw new InvalidBoardDescriptionException(string);
		}
		if (chars.hasNext()) {
			throw new InvalidBoardDescriptionException(string);
		} else {
			// all OK!
		}

		return board;
	}

	@SuppressWarnings("serial")
	public static class InvalidBoardDescriptionException extends RuntimeException {
		public InvalidBoardDescriptionException(String string) {
			super("The string representation of a board should be a 3x3 square of Xs, Os, and spaces:\n"
					+ formatBoardDescription(string));
		}
	}

	public static String formatBoardDescription(String string) {
		return string.trim().replaceAll(".*", "[$0]");
	}
}
