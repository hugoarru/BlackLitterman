# Black-Litterman Model Node.js/TypeScript Implementation

## Overview

This project is a Node.js/TypeScript implementation of the Black-Litterman model, proposed by Fischer Black and Robert Litterman.
This model integrates market equilibrium returns with investor's subjective views (market forecasts) to derive optimal asset allocation for a portfolio.

This implementation focuses on the core logic of the model: calculating equilibrium returns and computing posterior expected returns that incorporate investor views.

## Features

* Type-safe implementation using TypeScript
* Uses the `mathjs` library for matrix operations
* Designed for modern Node.js environments (ES modules)
* Supports investor views (absolute and relative views) with confidence levels

## Prerequisites

* Node.js (v20.x or later recommended; this project was developed and tested with v22.x)
* npm (included with Node.js)

## Installation

1.  **Clone the repository (if using a Git repository):**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install required dependencies:**
    ```bash
    npm install
    ```
    This will install the libraries listed in `package.json`, including `mathjs`, `ts-node`, and `typescript`.
    (If `package.json` is not provided, install them individually)
    ```bash
    npm install mathjs
    npm install --save-dev typescript ts-node @types/mathjs
    ```

## Configuration

### `tsconfig.json`

This project is configured to run in a modern Node.js (ESM) environment. The main `tsconfig.json` settings are as follows:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "strict": true,
    "skipLibCheck": true
  },
  "include": ["**/*.ts"],
  "exclude": ["node_modules"]
}
```

If `tsconfig.json` doesn't exist, generate it with the following command and configure it as shown above:

```bash
npx tsc --init
```

### Input Data (in `blackLitterman.ts`)

Model inputs are configured in the `exampleUsage` function within the `blackLitterman.ts` file. Modify the following parameters to match your actual data:

  * `marketCovariance` (`S`): Asset returns covariance matrix (N x N)
  * `marketCapWeights` (`w_mkt`): Market portfolio weights for each asset (N x 1)
  * `riskAversion` (`delta`): Market risk aversion coefficient
  * `P`: Investor view picking matrix (K x N)
  * `Q`: Expected returns corresponding to investor views (K x 1)
  * `Omega`: View error covariance matrix (K x K)
  * `tau`: Scalar indicating uncertainty of the prior distribution (equilibrium returns)

## Execution

Run the following command from the project root directory:

```bash
node --loader ts-node/esm blackLitterman.ts
```

Alternatively, you may also be able to run:

```bash
npx ts-node --esm blackLitterman.ts
```

Upon execution, the following information will be output to the console:

  * Input data
  * Equilibrium expected returns (Π)
  * Posterior expected returns (E[R])

## Code Structure

  * **`blackLitterman.ts`**: Contains the main model logic and execution example.
      * **Type Definitions:**
          * `Vector`: One-dimensional array of numbers.
          * `Matrix`: Two-dimensional array of numbers.
          * `BlackLittermanInputs`: Interface for model input parameters.
          * `BlackLittermanOutputs`: Interface for model outputs.
      * **Main Functions:**
          * `calculateEquilibriumReturns(...)`: Calculates equilibrium expected returns (Π).
          * `calculateBlackLittermanPosteriorReturns(...)`: Calculates posterior expected returns (E[R]) incorporating investor views.
          * `runBlackLitterman(...)`: Executes the above calculations and returns results.
          * `exampleUsage()`: Provides a concrete usage example with input data configuration.
      * **Matrix Operation Helpers:** Basic wrapper functions for matrix/vector operations using `mathjs`.

## License

This project does not specify a particular license. Feel free to use it as you wish.
(Add an MIT License or similar if needed.)

## Notes

  * This implementation provides the core computational part of the Black-Litterman model. When using it for actual investment decisions, a thorough understanding of input data accuracy and the model's assumptions and limitations is required.
  * Actual portfolio optimization (weight calculation) based on posterior expected returns is not included in this code. You will need to implement or use additional methods such as mean-variance optimization separately.
