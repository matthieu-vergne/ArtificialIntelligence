package fr.vergne.ai;

import java.util.Iterator;

public interface SensorMemory<Value> extends Iterable<Value> {
	public void add(Value value);
	public double getFamiliarity(Value value);
	@Override
	public Iterator<Value> iterator();
	public int size();
}
