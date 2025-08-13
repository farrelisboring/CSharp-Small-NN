class Program {
    static void Main(string[] args)
    {

        int[] labelData = new int[] { 0, 0, 0, 1 };
        int[,] inputData = new int[,]
        {
            { 0, 0 },
            { 0, 1 },
            { 1, 0 },
            { 1, 1 }
        };

        double learningRate;
        int epochNum;

        Perc perceptron = new Perc();
        perceptron.trainComp(labelData, learningRate = 0.05, inputData, epochNum = 50);
        Console.WriteLine("(0 & 0): " + perceptron.sigmoidComp(perceptron.predictComp(0, inputData)));
        Console.WriteLine("(0 & 1): " + perceptron.sigmoidComp(perceptron.predictComp(1, inputData)));
        Console.WriteLine("(1 & 0): " + perceptron.sigmoidComp(perceptron.predictComp(2, inputData)));
        Console.WriteLine("(1 & 1): " + perceptron.sigmoidComp(perceptron.predictComp(3, inputData)));
    }
}

class Perc
{

    private double[] weights = new double[] { 0.8, 0.7 };
    private double bias;

    public double predictComp(int inputRow, int[,] inputData) 
    {
        double taperFade = 0;
        taperFade = inputData[inputRow, 0] * weights[0] + inputData[inputRow, 1] * weights[1];
        return taperFade + bias;
    }


    public double sigmoidComp(double taperFade)
    {
        double compResult = 1.0 / (1.0 + Math.Exp(-taperFade));
        return compResult;
    }


    public void trainComp(int[] labelData, double learningRate, int[,] inputData, int epochsNum)
    {


        for (int epoch = 0; epoch < epochsNum; epoch++)
        {
            double[] batchGradients = new double[2];
            double batchErrors = 0;
            int n = labelData.Length;

            for (int r = 0; r < n; r++) 
            {
                double sampleError = sigmoidComp(predictComp(r, inputData)) - labelData[r];
                batchErrors += sampleError;
                for (int i = 0; i < 2; i++)
                {
                    batchGradients[i] += sampleError * inputData[r, i];
                }
            }
            
            weights[0] -= learningRate * batchGradients[0] / n;
            weights[1] -= learningRate * batchGradients[1] / n;
            bias -= learningRate * batchErrors / n;
            Console.WriteLine($"Epoch {epoch + 1}: Weights = [{weights[0]}, {weights[1]}], Bias = {bias}");
        }
        
    }
}