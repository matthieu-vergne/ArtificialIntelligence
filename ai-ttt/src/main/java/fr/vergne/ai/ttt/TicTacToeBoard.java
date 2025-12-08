package fr.vergne.ai.ttt;

public class TicTacToeBoard {

	public static enum Player {
		X, O
	}

	private final Player[][] board;

	public TicTacToeBoard() {
		board = new Player[3][3];
		reset();
	}

	public void set(int row, int col, Player player) {
		board[row][col] = player;
	}

	public Player get(int row, int col) {
		return board[row][col];
	}

	public boolean isFree(int row, int col) {
		return board[row][col] == null;
	}

	public void reset() {
		for (int row = 0; row < board.length; row++) {
			for (int col = 0; col < board[row].length; col++) {
				board[row][col] = null;
			}
		}
	}

}
