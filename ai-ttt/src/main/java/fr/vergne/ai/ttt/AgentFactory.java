package fr.vergne.ai.ttt;

import java.util.Random;

import fr.vergne.ai.agent.impl.ExperimentalAI;
import fr.vergne.ai.ttt.TicTacToeBoard.Player;

public class AgentFactory {

	public Runnable createExhaustiveAIPlayer(TicTacToeBoard board) {
		ExperimentalAI player = new ExperimentalAI();

		player.addSensor("R(0,0)", () -> board.get(0, 0));
		player.addSensor("R(0,1)", () -> board.get(0, 1));
		player.addSensor("R(0,2)", () -> board.get(0, 2));
		player.addSensor("R(1,0)", () -> board.get(1, 0));
		player.addSensor("R(1,1)", () -> board.get(1, 1));
		player.addSensor("R(1,2)", () -> board.get(1, 2));
		player.addSensor("R(2,0)", () -> board.get(2, 0));
		player.addSensor("R(2,1)", () -> board.get(2, 1));
		player.addSensor("R(2,2)", () -> board.get(2, 2));

		player.addActuator("WX(0,0)", () -> board.set(0, 0, Player.X));
		player.addActuator("WX(0,1)", () -> board.set(0, 1, Player.X));
		player.addActuator("WX(0,2)", () -> board.set(0, 2, Player.X));
		player.addActuator("WX(1,0)", () -> board.set(1, 0, Player.X));
		player.addActuator("WX(1,1)", () -> board.set(1, 1, Player.X));
		player.addActuator("WX(1,2)", () -> board.set(1, 2, Player.X));
		player.addActuator("WX(2,0)", () -> board.set(2, 0, Player.X));
		player.addActuator("WX(2,1)", () -> board.set(2, 1, Player.X));
		player.addActuator("WX(2,2)", () -> board.set(2, 2, Player.X));

		player.addActuator("WO(0,0)", () -> board.set(0, 0, Player.O));
		player.addActuator("WO(0,1)", () -> board.set(0, 1, Player.O));
		player.addActuator("WO(0,2)", () -> board.set(0, 2, Player.O));
		player.addActuator("WO(1,0)", () -> board.set(1, 0, Player.O));
		player.addActuator("WO(1,1)", () -> board.set(1, 1, Player.O));
		player.addActuator("WO(1,2)", () -> board.set(1, 2, Player.O));
		player.addActuator("WO(2,0)", () -> board.set(2, 0, Player.O));
		player.addActuator("WO(2,1)", () -> board.set(2, 1, Player.O));
		player.addActuator("WO(2,2)", () -> board.set(2, 2, Player.O));

		player.addActuator("reset", () -> board.reset());

		return player;
	}

	public Runnable createLimitedAIPlayer(TicTacToeBoard board) {
		ExperimentalAI player = new ExperimentalAI();
		int[] coords = { 0, 0 };

		player.addSensor("row", () -> coords[0]);
		player.addSensor("column", () -> coords[1]);
		player.addSensor("state", () -> board.get(coords[0], coords[1]));

		player.addActuator("go to row 0", () -> coords[0] = 0);
		player.addActuator("go to row 1", () -> coords[0] = 1);
		player.addActuator("go to row 2", () -> coords[0] = 2);
		player.addActuator("go to column 0", () -> coords[1] = 0);
		player.addActuator("go to column 1", () -> coords[1] = 1);
		player.addActuator("go to column 2", () -> coords[1] = 2);
		player.addActuator("play X", () -> board.set(coords[0], coords[1], Player.X));
		player.addActuator("play O", () -> board.set(coords[0], coords[1], Player.O));
		player.addActuator("reset", () -> board.reset());

		return player;
	}

	public Runnable createMinimalAIPlayer(TicTacToeBoard board) {
		ExperimentalAI player = new ExperimentalAI();
		int[] coords = { 0, 0 };
		Player[] state = { Player.X };

		player.addSensor("state", () -> board.get(coords[0], coords[1]));

		player.addActuator("next row", () -> coords[0] = (coords[0] + 1) % 3);
		player.addActuator("next column", () -> coords[1] = (coords[1] + 1) % 3);
		player.addActuator("next state", () -> state[0] = state[0] == Player.X ? Player.O : Player.X);
		player.addActuator("play", () -> board.set(coords[0], coords[1], state[0]));
		player.addActuator("reset", () -> board.reset());

		return player;
	}

	public Runnable createRandomPlayer(TicTacToeBoard board) {
		return new Runnable() {

			private final Random rand = new Random();

			@Override
			public void run() {
				int row = rand.nextInt(3);
				int col = rand.nextInt(3);
				Player state = Player.values()[rand.nextInt(2)];
				board.set(row, col, state);
			}
		};
	}
}
