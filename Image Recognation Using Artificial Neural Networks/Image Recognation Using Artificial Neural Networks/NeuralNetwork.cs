using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Schema;

namespace Image_Recognation_Using_Artificial_Neural_Networks
{
    internal class NeuralNetwork
    {
        Neuron N1;
        Neuron N2;
        public NeuralNetwork(int nRowSize, int nColSize)
        {
            Random rand = new Random();
            N1 = new Neuron(nRowSize, nColSize, rand);
            N2 = new Neuron(nRowSize, nColSize, rand);
        }
        public void TrainNeuron(int[,,] oneVariations, int[,,] twoVariations)
        {
            //Aşağıda eğitim için tüm clear input değerlerini hesaplatıyoruz.
            double[] n1_clear_one = N1.CalculateClearInputs(oneVariations);
            double[] n1_clear_two = N1.CalculateClearInputs(twoVariations);
            double[] n2_clear_one = N2.CalculateClearInputs(oneVariations);
            double[] n2_clear_two = N2.CalculateClearInputs(twoVariations);

            int epochNumber = 0; //Devir sayısı

            while (epochNumber < 40)
            {
                for (int i = 0; i < oneVariations.GetLength(0); i++) //Matrisleri döner
                {
                    if (n1_clear_one[i] <= n2_clear_one[i])
                    {
                        n1_clear_one[i] = N1.IncWeightAndClearInp(oneVariations, i);
                        n2_clear_one[i] = N2.DecWeightAndClearInp(oneVariations, i);
                    }
                    if (n2_clear_two[i] <= n1_clear_two[i])
                    {
                        n1_clear_two[i] = N1.DecWeightAndClearInp(twoVariations, i);
                        n2_clear_two[i] = N2.IncWeightAndClearInp(twoVariations, i);
                    }
                }
                epochNumber++;
            }
        }
        public int[] CalculateOutput(int[,,] matrix) //Verilen matrisin sonuçlarına bakmak için kullanılır.
        {
            //Nöronların ağırlık değerlerini alıyoruz.
            double[,] weightsN1 = N1.GetWeights();
            double[,] weightsN2 = N2.GetWeights();

            int[] estimatedValues = new int[matrix.GetLength(0)]; //Çıktılar bu listede tutulacak

            for (int k = 0; k < matrix.GetLength(0); k++) //Matrisleri döner
            {
                double clearInpN1 = 0;
                double clearInpN2 = 0;

                for (int l = 0; l < matrix.GetLength(1); l++)
                {
                    for (int m = 0; m < matrix.GetLength(2); m++)
                    {
                        clearInpN1 += matrix[k, l, m] * weightsN1[l, m];
                        clearInpN2 += matrix[k, l, m] * weightsN2[l, m];
                    }
                }

                if (clearInpN1 > clearInpN2)
                {
                    estimatedValues[k] = 1;
                }
                else if (clearInpN2 > clearInpN1)
                {
                    estimatedValues[k] = 2;
                }
            }
            return estimatedValues;
        }
    }
}
