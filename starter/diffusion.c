/*******************************************************************************
 * Diffusion Model
 *
 * A 2D rectangular world is filled with cells. Each of the four edges of the
 * world (top, left, right, and bottom) has its own unchanging value. The
 * middle of the world starts with an initial value, but as the model advances,
 * the value of each cell in the middle changes by setting itself equal to
 * the average of its four neighbors to the top, left, right, and bottom.
 * The model advances a certain number of time steps.
 *
 * The user can provide command line options in order to set the model
 * parameters. These are listed in the ParseArgs function.
 *
 * (c) Shodor, 2017
 ******************************************************************************/

/*************
 * Libraries *
 *************/
#include <omp.h> // For omp_get_wtime()
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/********************
 * Global variables *
 ********************/
int IsPrinting = true;
int NumRows = 10;
int NumCols = 10;
int NumSteps = 100;
float TopVal = 100.0;
float BottomVal = 0.0;
float LeftVal = 0.0;
float RightVal = 0.0;
float MiddleVal = 0.0;
int OutputPrecision = 3;
int NumCellsWithBounds;    // Calculated later using CalcNumCells()
int NumCellsWithoutBounds; // Calculated later using CalcNumCells()
int CellCharSize;          // Calculated later using CalcCellCharSize()
float * CellsWithBounds;   // Array of cell values, including the t,l,r,b bounds
float * CellsWithoutBounds; // Array of cell values, not including the bounds.
// We need 2 arrays because we do not want to "check" and "change" in the same
// loop; if a cell is updated before its neighbor checks its value, we will
// get the wrong result. See the CalcCells() and CopyCells() functions.
// We only need one of the arrays to have the cell boundaries; we can save
// memory with the other one by not including them.
// TODO - For each of the two arrays above, you will also need copies for the
// device memory

/**********************
 Function definitions *
 **********************/
// TODO - It is good practice to create a function to check whether a CUDA
// call was successful

// Check the command line arguments to see if the user provided any model
// parameters
void ParseArgs(int argc, char ** argv)
{
  char c;

  // See the documentation for getopt()
  while ((c = getopt(argc, argv, "qw:h:s:t:l:r:b:p:")) != -1)
  {
    switch(c)
    {
      case 'q':
        IsPrinting = false;
        break;
      case 'w':
        NumCols = atoi(optarg);
        break;
      case 'h':
        NumRows = atoi(optarg);
        break;
      case 's':
        NumSteps = atoi(optarg);
        break;
      case 't':
        TopVal = atof(optarg);
        break;
      case 'l':
        LeftVal = atof(optarg);
        break;
      case 'r':
        RightVal = atof(optarg);
        break;
      case 'b':
        BottomVal = atof(optarg);
        break;
      case 'p':
        OutputPrecision = atoi(optarg);
        break;
      case '?':
      default:
        fprintf(stderr, "Usage: %s [OPTIONS]\n"
          "OPTIONS:\n"
          "-q : turn off output\n"
          "-h <arg> : set number of rows\n"
          "-w <arg> : set number of columns\n"
          "-s <arg> : set number of time steps\n"
          "-t <arg> : set value of top boundary\n"
          "-l <arg> : set value of left boundary\n"
          "-r <arg> : set value of right boundary\n"
          "-b <arg> : set value of bottom boundary\n"
          "-p <arg> : set output precision\n",
          argv[0]);
    }
  }
}

// Calculate the number of cells in the world
void CalcNumCells()
{
  NumCellsWithBounds = (NumRows + 2) * (NumCols + 2);
  NumCellsWithoutBounds = NumRows * NumCols;
}

// Calculate the number of characters to use in displaying a single value
void CalcCellCharSize()
{
  int numDigits;

  // Initialize
  CellCharSize = 0;

  // If the top bound has the biggest value so far, use it
  if (TopVal > CellCharSize)
  {
    CellCharSize = TopVal;
  }

  // If the left bound has the biggest value so far, use it
  if (LeftVal > CellCharSize)
  {
    CellCharSize = LeftVal;
  }

  // If the right bound has the biggest value so far, use it
  if (RightVal > CellCharSize)
  {
    CellCharSize = RightVal;
  }

  // If the bottom bound has the biggest value so far, use it
  if (BottomVal > CellCharSize)
  {
    CellCharSize = BottomVal;
  }

  // Count the number of digits
  numDigits = 0;
  while (CellCharSize != 0)
  {
    CellCharSize /= 10;
    numDigits++;
  }
  CellCharSize = numDigits;

  // Add extra characters for the decimal point and the digits after it
  CellCharSize += 1 + OutputPrecision;
}

// Check if a call to malloc() was successful by examining its return value
void CheckMalloc(void * const val)
{
  if (val == NULL)
  {
    fprintf(stderr, "malloc error\n");
    exit(EXIT_FAILURE);
  }
}

// Allocate memory for dynamic arrays of cell values
void AllocMemory()
{
  CellsWithBounds    = (float *)malloc(NumCellsWithBounds * sizeof(float));
  CheckMalloc(CellsWithBounds);

  CellsWithoutBounds = (float *)malloc(NumCellsWithoutBounds *
    sizeof(float));
  CheckMalloc(CellsWithoutBounds);

  // TODO - This would be a good place to allocate memory on the device
}

// Set the initial cell values
void InitializeArrays()
{
  int row;
  int col;

  for (col = 0; col < NumCols; col++)
  {
    // Set the top bound
    CellsWithBounds[col + 1] = TopVal;

    // Set the bottom bound
    CellsWithBounds[(NumRows + 1) * (NumCols + 2) + col + 1] = BottomVal;
  }

  for (row = 0; row < NumRows; row++)
  {
    // Set the left bound
    CellsWithBounds[(row + 1) * (NumCols + 2) + 0] = LeftVal;

    // Set the middle
    for (col = 0; col < NumCols; col++)
    {
      CellsWithBounds[(row + 1) * (NumCols + 2) + col + 1] = MiddleVal;
    }

    // Set the right bound
    CellsWithBounds[(row + 1) * (NumCols + 2) + NumCols + 1] = LeftVal;
  }
}

// Print the values of each cell to the standard output
void PrintCells(int const time)
{
  int row;
  int col;

  printf("Time step %d\n", time);

  for (row = 0; row < NumRows; row++)
  {
    for (col = 0; col < NumCols; col++)
    {
      printf("%*.*f ", CellCharSize, OutputPrecision,
        CellsWithoutBounds[row * NumCols + col]);
    }
    printf("\n");
  }
  printf("\n");
}

// Calculate the average of nearest-neighbors for each cell. Make sure not
// to check and change values in the same array, or the calculation will be
// wrong.
void CalcCells()
{
  int row;
  int col;

  // TODO - This loop would be a good candidate for GPGPU parallelism
  for (row = 0; row < NumRows; row++)
  {
    for (col = 0; col < NumCols; col++)
    {
      // Start with nothing
      CellsWithoutBounds[row * NumCols + col] = 0.0;

      // Add the top neighbor
      CellsWithoutBounds[row * NumCols + col] +=
        CellsWithBounds[row * (NumCols + 2) + col + 1];

      // Add the left neighbor
      CellsWithoutBounds[row * NumCols + col] +=
        CellsWithBounds[(row + 1) * (NumCols + 2) + col];

      // Add the right neighbor
      CellsWithoutBounds[row * NumCols + col] +=
        CellsWithBounds[(row + 1) * (NumCols + 2) + col + 2];

      // Add the bottom neighbor
      CellsWithoutBounds[row * NumCols + col] +=
        CellsWithBounds[(row + 2) * (NumCols + 2) + col + 1];

      // Divide by 4 (multiplication is a bit faster, so multiply by 1/4)
      CellsWithoutBounds[row * NumCols + col] *= 0.25;
    }
  }
}

// Make sure both arrays have the new average of nearest-neighbors for each cell
void CopyCells()
{
  int row;
  int col;

  // TODO - This loop would be a good candidate for GPGPU parallelism
  for (row = 0; row < NumRows; row++)
  {
    for (col = 0; col < NumCols; col++)
    {
      CellsWithBounds[(row + 1) * (NumCols + 2) + col + 1] =
        CellsWithoutBounds[row * NumCols + col];
    }
  }
}

// Run the simulation
void Simulate()
{
  int time;

  // TODO - This would be a good place to copy data from the host to the device

  for (time = 0; time < NumSteps; time++)
  {
    if (IsPrinting)
    {
      // TODO - You will need to copy data from the device to the host here so
      // it can printed

      // Print the values of each cell to the standard output
      PrintCells(time);
    }

    // Calculate the average of nearest-neighbors for each cell. Make sure not
    // to check and change values in the same array, or the calculation will be
    // wrong.
    CalcCells();

    // Make sure both arrays have the new average of nearest-neighbors for each
    // cell
    CopyCells();
  }
}

// De-allocate memory for dynamic arrays of cell values
void FreeMemory()
{
  // TODO - This would be a good place to free the device memory

  free(CellsWithoutBounds);
  free(CellsWithBounds);
}

// The main function, where program execution starts
int main(int argc, char ** argv)
{
  // Start a timer
  double startTime = omp_get_wtime();

  // Check the command line arguments to see if the user provided any model
  // parameters
  ParseArgs(argc, argv);

  // Calculate the number of cells in the world
  CalcNumCells();

  // Calculate the number of characters to use in displaying a single value
  CalcCellCharSize();

  // Allocate memory for dynamic arrays of cell values
  AllocMemory();

  // Set the initial cell values
  InitializeArrays();

  // Run the simulation
  Simulate();

  // De-allocate memory for dynamic arrays of cell values
  FreeMemory();

  // Stop the timer
  printf("Runtime: %f seconds\n", omp_get_wtime() - startTime);
}

