package fr.vergne.ai;

public interface ArtificialIntelligence extends Runnable {

	public <T> void registerSensor(Sensor<T> sensor);

	public <T> void registerActuator(Actuator<T> actuator);
}
