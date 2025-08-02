class Program {
    static void Main(string[] args)
    {

        int[,] labelData = new int[,] {
            { 0 },
            { 0 },
            { 0 },
            { 1 }
        };

        double learningRate = 0.1;

        Perc perceptron = new Perc();
        perceptron.trainComp(labelData, learningRate);
        Console.WriteLine("(0, 0): " + perceptron.sigmoidComp(perceptron.predictComp(0)));
        Console.WriteLine("(0, 1): " + perceptron.sigmoidComp(perceptron.predictComp(1)));
        Console.WriteLine("(1, 0): " + perceptron.sigmoidComp(perceptron.predictComp(2)));
        Console.WriteLine("(1, 1): " + perceptron.sigmoidComp(perceptron.predictComp(3)));
    }
}

class Perc
{
    private int[,] inputData = new int[,]
    {
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 }
    };
    
    private double[] weights = new double[] { 0.8, 0.7 };
    private double bias;
    
    public double predictComp(int rowIndex)
    {
        double taperFade = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            taperFade += inputData[rowIndex, i] * weights[i];
        }
        return taperFade + bias;
    }

    public int sigmoidComp(double taperFade)
    {
        double compResult = 1.0 / (1.0 + Math.Exp(-taperFade));
        return compResult >= 0.5 ? 1 : 0;
    }

    public void trainComp(int[,] labelData, double learningRate, int epochsNum = 20) {

        for (int epoch = 0; epoch < epochsNum; epoch++) {
            for (int k = 0; k < 4; k++) {
                int predictLabel = sigmoidComp(predictComp(k));
                for (int r = 0; r < 2; r++) {
                    weights[r] += learningRate *  (labelData[k, 0] - predictLabel) * inputData[k,r];
                }
                bias += learningRate * (labelData[k, 0] - predictLabel);
            }
        }
    }
}