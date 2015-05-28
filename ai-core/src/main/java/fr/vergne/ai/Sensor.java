package fr.vergne.ai;

public interface Sensor<T> {

	public static interface SensorListener<T> {
		public void sensing(T value);
	}
	
	public void registerListener(SensorListener<T> listener);
}
