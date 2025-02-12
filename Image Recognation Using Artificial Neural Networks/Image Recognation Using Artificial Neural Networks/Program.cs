using System;
using System.Collections.Generic;
using System.Diagnostics.Tracing;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace Image_Recognation_Using_Artificial_Neural_Networks
{
    internal class Program
    {
        public int[,,] CreateOneVariations(int[,] matrix, int numVariations) //Orjinal bir matrisinden yola çıkarak varyasyonlar oluşturur ve döndürür.
                                                                             //numVariations = oluşturmak istenilen varyasyon sayısı
        {
            Random random = new Random();
            int rowsLength = matrix.GetLength(0);
            int colsLength = matrix.GetLength(1);

            int[,,] variationArray = new int[numVariations, rowsLength, colsLength]; //Oluşturulan varyassyonlar bu listeye atılacak

            for (int i = 0; i < numVariations; i++)
            {
                int[,] variation = (int[,])matrix.Clone();
                int numberPixelsToChange = random.Next(1, 6); //Matriste kaç tane pixelin değiştirileceği

                for (int j = 0; j < numberPixelsToChange; j++)
                {
                    int randomCol = random.Next(0, colsLength); //Matrisin son satırındaki değişecek pixelin indexi (sadece son satırda değiştirme işlemi yapılıyor)

                    if (matrix[rowsLength - 1, randomCol] == 0)
                    {
                        variation[rowsLength - 1, randomCol] = 1;
                    }
                    else
                    {
                        variation[rowsLength - 1, randomCol] = 0;
                    }

                    for (int k = 0; k < rowsLength; k++) //Varyasyonu listeye atama                    
                    {
                        for (int l = 0; l < colsLength; l++)
                        {
                            variationArray[i, k, l] = variation[k, l];
                        }
                    }
                }

            }
            return variationArray;
        }
        public int[,,] CreateTwoVariations(int[,] matrix, int numVariations) //Orjinal iki matrisinden yola çıkarak varyasyonlar oluşturur ve 
        {
            Random random = new Random();
            int rowsLength = matrix.GetLength(0);
            int colsLength = matrix.GetLength(1);

            int[,,] variationArray = new int[numVariations, rowsLength, colsLength]; //Oluşturulan varyassyonlar bu listeye atılacak

            for (int i = 0; i < numVariations; i++)
            {
                int[,] variation = (int[,])matrix.Clone();
                int numberPixelsToChange = random.Next(1, 5); //Matriste kaç tane pixelin değiştirileceği

                for (int j = 0; j < numberPixelsToChange; j++)
                {
                    int[] wantedIndexesToChange = { 0, 4 }; //Hem satırın hem sutunun sadece 1 ve 5. pixelleri değiştirilmek isteniyor
                    int numberColsToChange = wantedIndexesToChange[random.Next(0, wantedIndexesToChange.Length)];
                    int numberRowToChange = wantedIndexesToChange[random.Next(0, wantedIndexesToChange.Length)];

                    if (numberRowToChange == 0)
                    {
                        if (matrix[0, numberColsToChange] == 0)
                        {
                            variation[0, numberColsToChange] = 1;
                        }
                        else
                        {
                            variation[0, numberColsToChange] = 0;
                        }
                    }
                    if (numberRowToChange == rowsLength - 1)
                    {
                        if (matrix[rowsLength - 1, numberColsToChange] == 0)
                        {
                            variation[rowsLength - 1, numberColsToChange] = 1;
                        }
                        else
                        {
                            variation[rowsLength - 1, numberColsToChange] = 0;
                        }
                    }

                    for (int k = 0; k < rowsLength; k++)
                    {
                        for (int l = 0; l < colsLength; l++)
                        {
                            variationArray[i, k, l] = variation[k, l]; // matris arrayinin i'inci elemanına olusturulan matrisi atama
                        }
                    }


                }
            }


            return variationArray;
        }
        static void Main(string[] args)
        {
            // Orijinal 5x5 matrisler
            int[,] originalOne ={
            {0, 0, 1, 0, 0},
            {0, 1, 1, 0, 0},
            {0, 0, 1, 0, 0},
            {0, 0, 1, 0, 0},
            {0, 0, 1, 0, 0}
            };

            int[,] originalTwo = {
            {0, 1, 1, 1, 0},
            {1, 0, 0, 0, 1},
            {0, 0, 0, 1, 0},
            {0, 0, 1, 0, 0},
            {1, 1, 1, 1, 1}
            };

            Program main = new Program();
            int[,,] one_variations = main.CreateOneVariations(originalOne, 10);
            int[,,] two_variations = main.CreateTwoVariations(originalTwo, 10);

            int[] targetValues = new int[one_variations.GetLength(0) + two_variations.GetLength(0)]; //Matrislerimizin çıktı değerleri listesi (hedef değer)
            for (int i = 0; i < one_variations.GetLength(0); i++)
            {
                targetValues[i] = 1;
            }
            for (int j = one_variations.GetLength(0); j < targetValues.Length; j++)
            {
                targetValues[j] = 2;
            }

            NeuralNetwork neuralNetwork = new NeuralNetwork(5, 5);
            neuralNetwork.TrainNeuron(one_variations, two_variations);

            int[] oneOutputs = neuralNetwork.CalculateOutput(one_variations);
            int[] twoOutputs = neuralNetwork.CalculateOutput(two_variations);

            int[] estimatedValues = new int[targetValues.Length]; //Matrislerin output hesaplayıcısı ile hesaplanan çıktı değerleri listesi (tahminlenen değer)
            for (int i = 0; i < oneOutputs.Length; i++)
            {
                estimatedValues[i] = oneOutputs[i];
            }
            for (int j = oneOutputs.Length; j < targetValues.Length; j++)
            {
                estimatedValues[j] = twoOutputs[j - oneOutputs.Length];
            }

            // Tablo yazdırma
            Console.WriteLine("{0,-15} {1,-15}", "Hedef Değer", "Tahmin Değer");
            Console.WriteLine(new string('-', 30)); // Ayırıcı çizgi
            for (int i = 0; i < targetValues.Length; i++)
            {
                Console.WriteLine("{0,-15} {1,-15}", targetValues[i], estimatedValues[i]);
            }

            int numOfCorrectPredict = 0; //Doğru tahmin sayısı
            for (int i = 0; i < targetValues.Length; i++)
            {
                if (targetValues[i] == estimatedValues[i])
                {
                    numOfCorrectPredict++;
                }
            }
            double percentageOfAccuracy = ((double)numOfCorrectPredict / targetValues.Length) * 100;
            Console.WriteLine("Percentage of accuracy: " + percentageOfAccuracy + "%");

            int[,,] extraMatrices = new int[3, 5, 5] //Burada yazdığım matris değerleri sırayla 1,2,2
            {
                {
                    {0, 0, 1, 0, 0},
                    {0, 1, 1, 0, 0},
                    {0, 0, 1, 0, 0},
                    {0, 0, 1, 0, 0},
                    {1, 0, 0, 1, 0}
                },
                {
                    {1, 1, 1, 1, 0},
                    {1, 0, 0, 0, 1},
                    {0, 0, 0, 1, 0},
                    {0, 0, 1, 0, 0},
                    {1, 1, 1, 1, 0}
                },
                {
                    { 1, 1, 1, 1, 1},
                    { 1, 0, 0, 0, 1},
                    { 0, 0, 0, 1, 0},
                    { 0, 0, 1, 0, 0},
                    { 0, 1, 1, 1, 1}
                }
            };
            int[] estValForExrtaMatrices = new int[extraMatrices.GetLength(0)];
            estValForExrtaMatrices = neuralNetwork.CalculateOutput(extraMatrices);

            Console.WriteLine("************************************************");
            Console.WriteLine("Predictions from a Matrix Not Seen by the Model:");
            for (int i = 0; i < estValForExrtaMatrices.Length; i++)
            {
                Console.WriteLine(estValForExrtaMatrices[i]);
            }
        }
    }
}


