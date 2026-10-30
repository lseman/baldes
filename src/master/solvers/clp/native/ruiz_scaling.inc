      } else if (scalingMethod == 6) {
        // Simultaneous infinity-norm equilibration: apply the square root of
        // each inverse norm to the current row and column factors. Empty and
        // excluded columns do not influence the factors.
        std::vector<double> rowMax(numberRows);
        std::vector<double> columnMax(numberColumns);
        for (int pass = 0; pass < 5; ++pass) {
          std::fill(rowMax.begin(), rowMax.end(), 0.0);
          std::fill(columnMax.begin(), columnMax.end(), 0.0);
          for (iColumn = 0; iColumn < numberColumns; ++iColumn) {
            if (!usefulColumn[iColumn])
              continue;
            for (CoinBigIndex j = columnStart[iColumn];
                 j < columnStart[iColumn] + columnLength[iColumn]; ++j) {
              const int r = row[j];
              const double value = fabs(elementByColumn[j]) * rowScale[r] * columnScale[iColumn];
              rowMax[r] = CoinMax(rowMax[r], value);
              columnMax[iColumn] = CoinMax(columnMax[iColumn], value);
            }
          }
          for (iRow = 0; iRow < numberRows; ++iRow) {
            if (rowMax[iRow] > 1.0e-20)
              rowScale[iRow] = CoinMax(1.0e-10, CoinMin(1.0e10, rowScale[iRow] / sqrt(rowMax[iRow])));
          }
          for (iColumn = 0; iColumn < numberColumns; ++iColumn) {
            if (columnMax[iColumn] > 1.0e-20)
              columnScale[iColumn] = CoinMax(1.0e-10, CoinMin(1.0e10, columnScale[iColumn] / sqrt(columnMax[iColumn])));
          }
        }
