import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import java.util.Random;

public class TimeSeriesPrediction {

    private static final int NUM_KEY_PRODUCTS = 100;
    private static final int TRAIN_TIME_STEPS = 118;
    private static final int TEST_TIME_STEPS = 28;

    /**
     * Read time series data
     * @return An array containing the data where i,j-th entry equals the volume of product i at time j
     */
    public static int[][] readTimeSeries(){
        int[][] data = new int[NUM_KEY_PRODUCTS][120];
        for(int i=0; i<data.length; i++){
            Arrays.fill(data[i], 0);
        }
        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("product_distribution_training_set.txt")));
            String line;
            int key_prod_idx = 0;
            while((line=br.readLine())!=null){
                String[] tokens = line.trim().split("\\s+");
                for(int i=0; i<tokens.length; i++){
                    data[key_prod_idx][i] = Integer.parseInt(tokens[i]);
                }
                key_prod_idx++;

            }
            br.close();

        } catch(IOException e){
            e.printStackTrace();
        }
        return data;
    }

    /**
     * Build a sliding window model for predicting the time series
     * @param timeSeries the time series data
     * @param windowSize the window size for the sliding window model
     */
    public static void predict(int[][] timeSeries, int windowSize) throws IOException{
        ArrayList<Attribute> fvWekaAttributes = new ArrayList<Attribute>();
        double tot_error = 0;
        for(int i=0; i<windowSize; i++) {
            Attribute attr = new Attribute("time_"+i);
            fvWekaAttributes.add(attr);
        }
        double[] total = new double[TEST_TIME_STEPS];
		List<String> output = new ArrayList<String>();

		for (int i = 0; i < NUM_KEY_PRODUCTS; i++) {
		    // create single task model for each product
		    // then make predictions for that product
		    Instances trainingset = new Instances("Relation", fvWekaAttributes, TRAIN_TIME_STEPS - windowSize);
		    trainingset.setClassIndex(windowSize - 1);
		    for (int j = 1; j < TRAIN_TIME_STEPS - windowSize; j++) {
		        double[] values = new double[windowSize];
		        for (int k = j; k < j + windowSize; k++) {
		            values[k - j] = timeSeries[i][k];
		        }
		        Instance slot = new DenseInstance(1.0, values);
		        trainingset.add(slot);
		    }
		    // evaluation
		    try {
		        Evaluation eval = new Evaluation(trainingset);
		        eval.crossValidateModel(new RandomForest(), trainingset, 5, new Random(1));
		        double rmse = eval.rootMeanSquaredError();
		        tot_error += rmse;
		        System.out.println("RMSE for model for key product " + timeSeries[i][0] + ": " + rmse);
		    } catch (Exception e) {
		        e.printStackTrace();
		    }
		    // build model on training data and apply to test data
		    // and output predictions
		    try {
		        RandomForest rf = new RandomForest();
		        rf.buildClassifier(trainingset);

		        double[] predictions = new double[TEST_TIME_STEPS];
		        for (int j = 0; j < TEST_TIME_STEPS; j++) {
		            double[] values = new double[windowSize];
		            int k = 0;
		            for (; k < windowSize - j - 1; k++) {
		                values[k] = timeSeries[i][119 - windowSize+ k];
		            }

		            int surplus = j-windowSize+1;

		            for (; k < windowSize - 1; k++) {
		                values[k] = predictions[k+surplus];
		            }

		            Instances instances = new Instances("Relation", fvWekaAttributes, 1);
		            instances.setClassIndex(windowSize - 1);
		            instances.add(new DenseInstance(1.0, values));
		            double prediction = rf.classifyInstance(instances.firstInstance());

		            predictions[j] = prediction;
		        }

		        String out = "";

		        out += "" + timeSeries[i][0];
		        for (int j = 0; j < TEST_TIME_STEPS; j++) {
		            total[j] += predictions[j];
		            out += " " + Math.round(predictions[j]);
		        }
		        out += "\n";

		        output.add(out);

		    } catch (Exception e) {
		        e.printStackTrace();
		    }
		}
		String out = "";
		out += "" + 0;
		for (int j = 0; j < TEST_TIME_STEPS; j++) {
		    out += " " + Math.round(total[j]);
		}
		out += "\n";
		output.add(0, out);

		 try {
	            BufferedWriter bw = new BufferedWriter(new FileWriter(new File("prediction.txt")));
	            for(int i=0; i<output.size(); i++){
	                bw.write(output.get(i));
	            }
	            bw.close();
	        } catch (IOException e) {
	            e.printStackTrace();
	        }
			System.out.println("");
	        System.out.println(" Mean RMSE: " + tot_error / 100.0);
	        System.out.println(" Results written to ---> prediction.txt");
    }

    public static void main(String[] args) throws IOException{
        int[][] data = readTimeSeries();
        predict(data, 14);
    }
}
