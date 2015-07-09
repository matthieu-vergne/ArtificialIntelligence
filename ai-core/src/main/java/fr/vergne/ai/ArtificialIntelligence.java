package fr.vergne.ai;

public interface ArtificialIntelligence extends Runnable {

	public <Value> void addSensor(Sensor<Value> sensor);

	public void addActuator(Actuator actuator);
}
