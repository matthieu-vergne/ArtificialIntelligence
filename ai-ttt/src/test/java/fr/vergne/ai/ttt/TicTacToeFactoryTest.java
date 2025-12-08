package fr.vergne.ai.ttt;

import static org.junit.Assert.*;

import org.junit.Test;

import fr.vergne.ai.ttt.TicTacToeBoard.Player;
import fr.vergne.ai.ttt.TicTacToeFactory.InvalidBoardDescriptionException;

public class TicTacToeFactoryTest {

	@Test
	public void testCreate3x3Board() {
		TicTacToeFactory factory = new TicTacToeFactory();
		TicTacToeBoard board = factory.createBoard("xox\nox \n xo");
		assertEquals(Player.X, board.get(0, 0));
		assertEquals(Player.O, board.get(0, 1));
		assertEquals(Player.X, board.get(0, 2));
		assertEquals(Player.O, board.get(1, 0));
		assertEquals(Player.X, board.get(1, 1));
		assertEquals(null, board.get(1, 2));
		assertEquals(null, board.get(2, 0));
		assertEquals(Player.X, board.get(2, 1));
		assertEquals(Player.O, board.get(2, 2));
	}

	@Test
	public void testInvalidCharacterThrowsException() {
		TicTacToeFactory factory = new TicTacToeFactory();
		try {
			factory.createBoard("axx\nxxx\nxxx");
			fail("No exception thrown");
		} catch (InvalidBoardDescriptionException e) {
			// OK
		}
		try {
			factory.createBoard("xxx\nxbx\nxxx");
			fail("No exception thrown");
		} catch (InvalidBoardDescriptionException e) {
			// OK
		}
		try {
			factory.createBoard("xxx\nxxx\nxxc");
			fail("No exception thrown");
		} catch (InvalidBoardDescriptionException e) {
			// OK
		}
	}

	@Test
	public void testInconsistentSizeThrowsException() {
		TicTacToeFactory factory = new TicTacToeFactory();
		try {
			factory.createBoard("xxxxxxxx\nxxx\nxxx");
			fail("No exception thrown");
		} catch (InvalidBoardDescriptionException e) {
			// OK
		}
		try {
			factory.createBoard("xxx\nxxxxxxxx\nxxx");
			fail("No exception thrown");
		} catch (InvalidBoardDescriptionException e) {
			// OK
		}
		try {
			factory.createBoard("xxx\nxxx\nxxxxxxxx");
			fail("No exception thrown");
		} catch (InvalidBoardDescriptionException e) {
			// OK
		}
	}

}
