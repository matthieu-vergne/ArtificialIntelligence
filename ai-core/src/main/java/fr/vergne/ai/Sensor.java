package fr.vergne.ai;

public interface Sensor<Value> {

	public static interface SensorListener<Value> {
		public void sensing(Value value);
	}

	public void addSensorListener(SensorListener<Value> listener);

	public void removeSensorListener(SensorListener<Value> listener);
}
