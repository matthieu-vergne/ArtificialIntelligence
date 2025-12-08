package fr.vergne.ai.ttt;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.logging.LogManager;

import fr.vergne.ai.ttt.TicTacToeBoard.Player;

/**
 * The {@link TicTacToeEnvironment} aims at providing an environment in which
 * Tic-Tac-Toe artefacts can evolve. In particular, players can use freely a
 * {@link TicTacToeBoard}, or they can follow the orders of a
 * {@link TicTacToeManager} to run an actual game.
 * 
 * @author Matthieu Vergne <matthieu.vergne@gmail.com>
 *
 */
public class TicTacToeEnvironment {

	public static void main(String[] args) {
		try {
			FileInputStream configFile = new FileInputStream("logging.properties");
			LogManager.getLogManager().readConfiguration(configFile);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		TicTacToeBoard board = new TicTacToeBoard();

		AgentFactory agentFactory = new AgentFactory();
		// TODO Try various AI profiles
		Runnable playerX = agentFactory.createExhaustiveAIPlayer(board);

		RunnableFactory runnableFactory = new RunnableFactory();
		launch(runnableFactory.createInfiniteLoop(playerX));

		TicTacToeManager manager = new TicTacToeManager();
		manager.setBoard(board);
		manager.addPlayer(playerX);
		manager.runTicTacToeGame();
	}

	private static void launch(Runnable player) {
		Thread thread = new Thread(player);
		thread.setDaemon(true);
		thread.start();
	}
}
