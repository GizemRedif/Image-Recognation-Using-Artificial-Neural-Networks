using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Image_Recognation_Using_Artificial_Neural_Networks
{
    internal class Neuron
    {
        private double[,] initialWeights; //Rastgele belirlenecek başlangıç ağırlıkları
        double learningCoefficient = 0.001; //lambda/öğrenme katsayısı
        public Neuron(int rowSize, int colSize, Random rand) //Neuron için initial weightleri oluşturur
        {
            initialWeights = new double[rowSize, colSize];

            for (int i = 0; i < rowSize; i++)
            {
                for (int j = 0; j < colSize; j++)
                {
                    initialWeights[i, j] = rand.NextDouble() * 2 - 1; //Değerlerimiz -1 ile 1 arasında rastgele oluşur.
                }

            }
        }
        public double[] CalculateClearInputs(int[,,] variations) //Sırayla her matris(varyasyon) için "initial weight" ve "input" değerlerinin çarpılarak toplanmış net girdisini hessaplar
        {
            double[] clearInputs = new double[variations.GetLength(0)];

            for (int k = 0; k < variations.GetLength(0); k++)
            {
                double clearInput = 0; //Bir matris icin initialWeight*input degerlerinin toplamı
                for (int l = 0; l < variations.GetLength(1); l++)
                {
                    for (int m = 0; m < variations.GetLength(2); m++)
                    {
                        clearInput += variations[k, l, m] * initialWeights[l, m];
                    }

                }
                clearInputs[k] = clearInput;
            }
            return clearInputs;
        }
        public double IncWeightAndClearInp(int[,,] variationList, int matrixToBeChanged) //Verilen nöronun weight değerlerini arttırır ve clear inputu tekrar hesaplar.
        {
            double clearInput = 0;
            for (int i = 0; i < variationList.GetLength(1); i++)
            {
                for (int j = 0; j < variationList.GetLength(2); j++)
                {
                    initialWeights[i, j] += learningCoefficient * variationList[matrixToBeChanged, i, j]; //weight değeri öğrenme katsayı ve input değerinin çarpımı kadar arttırılır.
                    clearInput += variationList[matrixToBeChanged, i, j] * initialWeights[i, j];
                }
            }
            return clearInput;
        }
        public double DecWeightAndClearInp(int[,,] variationList, int matrixToBeChanged) //Verilen nöronun weight değerlerini azaltır ve clear inputu tekrar hesaplar.
        {
            double clearInput = 0;
            for (int i = 0; i < variationList.GetLength(1); i++)
            {
                for (int j = 0; j < variationList.GetLength(2); j++)
                {
                    initialWeights[i, j] -= learningCoefficient * variationList[matrixToBeChanged, i, j]; //weight değeri öğrenme katsayı ve input değerinin çarpımı kadar azaltılır.
                    clearInput += variationList[matrixToBeChanged, i, j] * initialWeights[i, j];
                }
            }
            return clearInput;
        }
        public double[,] GetWeights() //Nöronun weight değerlerini liste olarak döndürür.
        {
            double[,] weights = new double[initialWeights.GetLongLength(0), initialWeights.GetLength(1)];

            for (int i = 0; i < initialWeights.GetLength(0); i++)
            {
                for (int j = 0; j < initialWeights.GetLength(1); j++)
                {
                    weights[i, j] = initialWeights[i, j];
                }
            }
            return weights;
        }
    }
}