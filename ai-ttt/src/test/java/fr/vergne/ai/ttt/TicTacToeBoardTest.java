package fr.vergne.ai.ttt;

import static org.junit.Assert.*;

import org.junit.Test;

import fr.vergne.ai.ttt.TicTacToeBoard.Player;

public class TicTacToeBoardTest {

	@Test
	public void testNewInstanceIsEmpty() {
		TicTacToeBoard board = new TicTacToeBoard();
		for (int row = 0; row < 3; row++) {
			for (int col = 0; col < 3; col++) {
				assertEquals("(" + row + "," + col + ")", null, board.get(row, col));
			}
		}
	}

	@Test
	public void testSetGetState() {
		TicTacToeBoard board = new TicTacToeBoard();

		board.set(1, 2, Player.X);
		assertEquals(Player.X, board.get(1, 2));

		board.set(2, 0, Player.O);
		assertEquals(Player.X, board.get(1, 2));
		assertEquals(Player.O, board.get(2, 0));

		board.set(1, 1, Player.X);
		assertEquals(Player.X, board.get(1, 2));
		assertEquals(Player.O, board.get(2, 0));
		assertEquals(Player.X, board.get(1, 1));
	}

	@Test
	public void testIsFree() {
		TicTacToeBoard board = new TicTacToeBoard();

		// Check test validity
		assertTrue(board.isFree(1, 2));
		assertTrue(board.isFree(2, 0));
		assertTrue(board.isFree(1, 1));

		board.set(1, 2, Player.X);
		assertFalse(board.isFree(1, 2));
		assertTrue(board.isFree(2, 0));
		assertTrue(board.isFree(1, 1));

		board.set(2, 0, Player.O);
		assertFalse(board.isFree(1, 2));
		assertFalse(board.isFree(2, 0));
		assertTrue(board.isFree(1, 1));

		board.set(1, 1, Player.X);
		assertFalse(board.isFree(1, 2));
		assertFalse(board.isFree(2, 0));
		assertFalse(board.isFree(1, 1));
	}

	@Test
	public void testResetMakesEverythingFree() {
		TicTacToeBoard board = new TicTacToeBoard();
		board.set(0, 1, Player.X);
		board.set(1, 1, Player.O);
		board.set(2, 0, Player.O);
		board.set(2, 2, Player.X);

		board.reset();
		for (int row = 0; row < 3; row++) {
			for (int col = 0; col < 3; col++) {
				assertTrue("(" + row + "," + col + ")", board.isFree(row, col));
			}
		}
	}

}
