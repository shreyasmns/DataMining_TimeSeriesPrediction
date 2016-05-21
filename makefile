all: main

main:
	javac -classpath weka.jar TimeSeriesPrediction.java
	java -classpath weka.jar:. TimeSeriesPrediction

clean:
	rm *.class
	