package fr.vergne.ai.ttt;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Logger;

import fr.vergne.ai.agent.impl.ExperimentalAI;
import fr.vergne.ai.ttt.TicTacToeBoard.Player;

/**
 * A {@link TicTacToeManager} aims at managing classic Tic-Tac-Toe games. While
 * a {@link TicTacToeBoard} provides the support of the game, it does not
 * enforce any rule and can be used freely by any player. This is the role of
 * the {@link TicTacToeManager} to enforce them.
 * 
 * @author Matthieu Vergne <matthieu.vergne@gmail.com>
 *
 */
public class TicTacToeManager {

	final Logger logger = Logger.getLogger(ExperimentalAI.class.getName());

	public static enum Winner {
		UNKNOWN, DRAW, X, O
	}

	private TicTacToeBoard board;
	private final Collection<Runnable> players = new HashSet<>();

	public void setBoard(TicTacToeBoard board) {
		if (board == null) {
			throw new NullPointerException("Invalid board");
		} else {
			this.board = board;
		}
	}

	public TicTacToeBoard getBoard() {
		return board;
	}

	public void addPlayer(Runnable player) {
		if (player == null) {
			throw new NullPointerException("Invalid player");
		} else {
			players.add(player);
		}
	}

	public void runTicTacToeGame() {
		Player currentPlayer = Player.X;
		Runnable playerX;
		Runnable playerO;
		if (players.size() == 0) {
			throw new IllegalStateException("No player to play with.");
		} else if (players.size() == 1) {
			playerX = players.iterator().next();
			// TODO Auto-play player O
			playerO = new AgentFactory().createRandomPlayer(board);
		} else if (players.size() >= 2) {
			List<Runnable> remaining = new LinkedList<>(players);
			Collections.shuffle(remaining);
			Iterator<Runnable> iterator = remaining.iterator();
			playerX = iterator.next();
			playerO = iterator.next();
		} else {
			throw new RuntimeException("This case should not happen");
		}
		while (getWinner(board) == Winner.UNKNOWN) {
			if (currentPlayer == Player.X) {
				playerX.run();
			} else {
				playerO.run();
			}
		}
	}

	private Winner getWinner(TicTacToeBoard board) {
		if (hasWinPattern(board, Player.X)) {
			return Winner.X;
		} else if (hasWinPattern(board, Player.O)) {
			return Winner.O;
		} else if (hasFreePosition(board)) {
			return Winner.UNKNOWN;
		} else {
			return Winner.DRAW;
		}
	}

	private boolean hasFreePosition(TicTacToeBoard board) {
		for (int row = 0; row < 3; row++) {
			for (int col = 0; col < 3; col++) {
				if (board.isFree(row, col)) {
					return true;
				} else {
					continue;
				}
			}
		}
		return false;
	}

	private boolean hasWinPattern(TicTacToeBoard board, Player state) {
		return isRowFilled(board, 0, state) || isRowFilled(board, 1, state) || isRowFilled(board, 2, state)
				|| isColFilled(board, 0, state) || isColFilled(board, 1, state) || isColFilled(board, 2, state)
				|| isDiagonal1Filled(board, state) || isDiagonal2Filled(board, state);
	}

	private boolean isRowFilled(TicTacToeBoard board, int row, Player state) {
		return board.get(row, 0) == state && board.get(row, 1) == state && board.get(row, 2) == state;
	}

	private boolean isColFilled(TicTacToeBoard board, int col, Player state) {
		return board.get(0, col) == state && board.get(1, col) == state && board.get(2, col) == state;
	}

	private boolean isDiagonal1Filled(TicTacToeBoard board, Player state) {
		return board.get(0, 0) == state && board.get(1, 1) == state && board.get(2, 2) == state;
	}

	private boolean isDiagonal2Filled(TicTacToeBoard board, Player state) {
		return board.get(0, 2) == state && board.get(1, 1) == state && board.get(2, 0) == state;
	}
}
