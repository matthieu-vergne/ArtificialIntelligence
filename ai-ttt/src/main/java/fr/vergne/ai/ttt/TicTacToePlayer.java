package fr.vergne.ai.ttt;

public interface TicTacToePlayer {
	public void setGoal(/* TODO context */);
	
	public void informSuccess(float success);
	
	public void informFailure(float failure);
}
