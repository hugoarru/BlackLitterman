import * as math from 'mathjs';

// --- Type Definitions ---
type Vector = number[];
type Matrix = number[][];

/**
 * Input for the Black-Litterman model
 */
interface BlackLittermanInputs {
    /** tau: Scalar. Uncertainty of the prior distribution (equilibrium returns). Smaller values indicate higher confidence in the prior. (e.g., 0.025) */
    tau: number;
    /** P: K x N matrix. Investor view picking matrix. K is the number of views, N is the number of assets. */
    P: Matrix;
    /** Q: K x 1 vector. Expected returns corresponding to investor views. */
    Q: Vector;
    /** Omega: K x K matrix. View error covariance matrix (usually diagonal). Smaller diagonal elements indicate higher confidence in views. */
    Omega: Matrix;
    /** marketCovariance: N x N matrix (S). Asset returns covariance matrix. */
    marketCovariance: Matrix;
    /** marketCapWeights: N x 1 vector (w_mkt). Market portfolio weights for each asset. */
    marketCapWeights: Vector;
    /** riskAversion: Scalar (delta). Market risk aversion coefficient. */
    riskAversion: number;
}

/**
 * Output from the Black-Litterman model
 */
interface BlackLittermanOutputs {
    /** posteriorReturns: N x 1 vector. Posterior expected returns incorporating views. */
    posteriorReturns: Vector;
    /** posteriorCovariance: N x N matrix. Posterior covariance matrix incorporating views (here we return the prior covariance for simplicity, but more complex calculations are possible). */
    posteriorCovariance: Matrix;
    /** equilibriumReturns: N x 1 vector. Equilibrium expected returns. */
    equilibriumReturns: Vector;
}

// --- Matrix/Vector Operation Helper Functions (using mathjs) ---

function transpose(matrix: Matrix): Matrix {
    return math.transpose(matrix) as Matrix;
}

function multiply(A: math.MathType, B: math.MathType): math.MathType {
    return math.multiply(A, B);
}

function inverse(matrix: Matrix): Matrix {
    try {
        return math.inv(matrix) as Matrix;
    } catch (e) {
        console.error("Matrix inversion failed. The matrix might be singular.", matrix);
        throw new Error(`Matrix inversion failed: ${e}`);
    }
}

function add(A: math.MathType, B: math.MathType): math.MathType {
    return math.add(A, B);
}

function subtract(A: math.MathType, B: math.MathType): math.MathType {
    return math.subtract(A, B);
}

// Convert vector to column vector (N x 1 matrix)
function toColumnMatrix(vector: Vector): Matrix {
    return vector.map(v => [v]);
}

// Convert N x 1 matrix to vector
function toVector(matrix: Matrix): Vector {
    return matrix.map(row => row[0]);
}


// --- Black-Litterman Model Core Logic ---

/**
 * Calculates equilibrium expected returns (Implied Equilibrium Returns).
 * Π = δ * S * w_mkt
 * @param riskAversion Risk aversion coefficient (δ)
 * @param S Market covariance matrix (N x N)
 * @param wMkt Market portfolio weights (N x 1)
 * @returns Equilibrium expected returns vector (N x 1)
 */
function calculateEquilibriumReturns(
    riskAversion: number,
    S: Matrix,
    wMkt: Vector
): Vector {
    if (S.length === 0 || S[0].length !== wMkt.length) {
        throw new Error("Dimension mismatch: marketCovariance (S) and marketCapWeights (wMkt).");
    }
    if (S.length !== S[0].length) {
        throw new Error("marketCovariance (S) must be a square matrix.");
    }

    const wMktCol = toColumnMatrix(wMkt); // N x 1
    const S_times_wMkt = multiply(S, wMktCol) as Matrix; // (N x N) * (N x 1) = N x 1
    const equilibriumReturnsMatrix = multiply(riskAversion, S_times_wMkt) as Matrix; // N x 1

    return toVector(equilibriumReturnsMatrix);
}

/**
 * Calculates Black-Litterman posterior expected returns (Posterior Expected Returns).
 * E[R] = Π + (τS)Pᵀ[P(τS)Pᵀ + Ω]⁻¹(Q - PΠ)
 *
 * @param tau Scalar indicating uncertainty of the prior distribution
 * @param S Market covariance matrix (N x N)
 * @param Pi_eq Equilibrium expected returns vector (N x 1)
 * @param P View picking matrix (K x N)
 * @param Q View expected returns vector (K x 1)
 * @param Omega View error covariance matrix (K x K)
 * @returns Posterior expected returns vector (N x 1)
 */
function calculateBlackLittermanPosteriorReturns(
    tau: number,
    S: Matrix,
    Pi_eq: Vector,
    P: Matrix,
    Q: Vector,
    Omega: Matrix
): Vector {
    const N = S.length; // Number of assets
    const K = Q.length; // Number of views

    if (P.length !== K || (P[0] && P[0].length !== N)) {
        throw new Error("Dimension mismatch for P matrix.");
    }
    if (Omega.length !== K || (Omega[0] && Omega[0].length !== K)) {
        throw new Error("Dimension mismatch for Omega matrix.");
    }
    if (Pi_eq.length !== N) {
        throw new Error("Dimension mismatch for equilibrium returns Pi_eq.");
    }

    const Pi_eq_col = toColumnMatrix(Pi_eq); // N x 1
    const Q_col = toColumnMatrix(Q);       // K x 1

    // τ * S
    const tauS: Matrix = multiply(tau, S) as Matrix; // N x N

    // P * Π
    const P_Pi = multiply(P, Pi_eq_col) as Matrix; // (K x N) * (N x 1) = K x 1

    // Q - P * Π (view excess returns)
    const Q_minus_P_Pi = subtract(Q_col, P_Pi) as Matrix; // K x 1

    // P * (τS)
    const P_tauS = multiply(P, tauS) as Matrix; // (K x N) * (N x N) = K x N

    // P * (τS) * Pᵀ
    const P_tauS_PT = multiply(P_tauS, transpose(P)) as Matrix; // (K x N) * (N x K) = K x K

    // P * (τS) * Pᵀ + Ω
    const bracket_term_inv_target = add(P_tauS_PT, Omega) as Matrix; // K x K

    // [P * (τS) * Pᵀ + Ω]⁻¹
    const bracket_term_inverted = inverse(bracket_term_inv_target); // K x K

    // (τS) * Pᵀ
    const tauS_PT = multiply(tauS, transpose(P)) as Matrix; // (N x N) * (N x K) = N x K

    // (τS)Pᵀ[P(τS)Pᵀ + Ω]⁻¹
    const right_multiplier = multiply(tauS_PT, bracket_term_inverted) as Matrix; // (N x K) * (K x K) = N x K

    // (τS)Pᵀ[P(τS)Pᵀ + Ω]⁻¹(Q - PΠ)
    const adjustment_matrix = multiply(right_multiplier, Q_minus_P_Pi) as Matrix; // (N x K) * (K x 1) = N x 1

    // E[R] = Π + adjustment
    const posteriorReturnsMatrix = add(Pi_eq_col, adjustment_matrix) as Matrix; // N x 1

    return toVector(posteriorReturnsMatrix);
}

// --- Main Function ---
/**
 * Runs the Black-Litterman model.
 * @param inputs Black-Litterman model inputs
 * @returns Black-Litterman model outputs
 */
function runBlackLitterman(inputs: BlackLittermanInputs): BlackLittermanOutputs {
    const {
        tau,
        P,
        Q,
        Omega,
        marketCovariance,
        marketCapWeights,
        riskAversion,
    } = inputs;

    // 1. Calculate equilibrium expected returns (Π)
    const equilibriumReturns = calculateEquilibriumReturns(
        riskAversion,
        marketCovariance,
        marketCapWeights
    );

    // 2. Calculate posterior expected returns (E[R])
    const posteriorReturns = calculateBlackLittermanPosteriorReturns(
        tau,
        marketCovariance,
        equilibriumReturns,
        P,
        Q,
        Omega
    );

    // Note: The posterior covariance matrix in the Black-Litterman model can also be derived.
    // M_posterior_inv = (tau*S)^-1 + P^T * Omega^-1 * P
    // M_posterior = (M_posterior_inv)^-1
    // For portfolio optimization, use this posterior covariance matrix, or simply the original market covariance matrix (S).
    // Here, for simplicity, we return the original marketCovariance, but calculate posterior covariance if needed.
    // const tauS_inv = inverse(multiply(tau, marketCovariance) as Matrix);
    // const PT_Omega_inv = multiply(transpose(P), inverse(Omega)) as Matrix;
    // const PT_Omega_inv_P = multiply(PT_Omega_inv, P) as Matrix;
    // const M_posterior_inv = add(tauS_inv, PT_Omega_inv_P) as Matrix;
    // const posteriorCovariance = inverse(M_posterior_inv);

    return {
        posteriorReturns,
        posteriorCovariance: marketCovariance, // Or use the calculated posteriorCovariance above
        equilibriumReturns,
    };
}

// --- Usage Example ---
function exampleUsage() {
    console.log("--- Black-Litterman Model Execution Example ---");

    // Example: 3 assets, 2 views
    const numAssets = 3;
    const numViews = 2;

    // Market data
    const marketCovariance: Matrix = [ // S (NxN)
        [0.0225, 0.0068, 0.0033], // e.g., [Asset1_Var, Asset1_Asset2_Cov, Asset1_Asset3_Cov]
        [0.0068, 0.0289, 0.0078], // e.g., [Asset2_Asset1_Cov, Asset2_Var, Asset2_Asset3_Cov]
        [0.0033, 0.0078, 0.0441], // e.g., [Asset3_Asset1_Cov, Asset3_Asset2_Cov, Asset3_Var]
    ];
    const marketCapWeights: Vector = [0.5, 0.3, 0.2]; // w_mkt (Nx1) - Total 100%
    const riskAversion: number = 3.0; // delta - Risk aversion coefficient

    // Investor views
    // P (KxN): Each row represents one view.
    // View 1: Asset 1 outperforms Asset 2 by 1% (Asset1 - Asset2 = 0.01)
    // View 2: Asset 3 has an absolute return of 4% (Asset3 = 0.04)
    const P: Matrix = [
        [1, -1, 0],
        [0,  0, 1],
    ];
    const Q: Vector = [0.01, 0.04]; // Q (Kx1): Returns corresponding to views

    // Omega (KxK): View uncertainty (variance). Smaller diagonal elements indicate higher confidence.
    // Usually a diagonal matrix. Off-diagonal elements represent correlations between views.
    const view1Variance = 0.0001; // View 1 error variance (std dev 0.01)
    const view2Variance = 0.0005; // View 2 error variance (std dev approx 0.022)
    const Omega: Matrix = [
        [view1Variance, 0],
        [0, view2Variance],
    ];

    const tau: number = 0.05; // Scalar for prior distribution uncertainty

    const inputs: BlackLittermanInputs = {
        tau,
        P,
        Q,
        Omega,
        marketCovariance,
        marketCapWeights,
        riskAversion,
    };

    try {
        const outputs = runBlackLitterman(inputs);

        console.log("\nInput Data:");
        console.log("Tau:", tau);
        console.log("P (View Matrix):"); P.forEach(row => console.log(row));
        console.log("Q (View Returns):", Q);
        console.log("Omega (View Error Covariance):"); Omega.forEach(row => console.log(row));
        console.log("Market Covariance Matrix S:"); marketCovariance.forEach(row => console.log(row));
        console.log("Market Weights w_mkt:", marketCapWeights);
        console.log("Risk Aversion Coefficient delta:", riskAversion);

        console.log("\n--- Model Output ---");
        console.log("Equilibrium Expected Returns (Π):");
        outputs.equilibriumReturns.forEach((ret, i) => console.log(`  Asset ${i+1}: ${(ret * 100).toFixed(3)}%`));

        console.log("\nPosterior Expected Returns (E[R]):");
        outputs.posteriorReturns.forEach((ret, i) => console.log(`  Asset ${i+1}: ${(ret * 100).toFixed(3)}%`));

        // console.log("\nPosterior Covariance Matrix:"); // Display if needed
        // outputs.posteriorCovariance.forEach(row => console.log(row.map(val => val.toFixed(5))));

        console.log("\n--- Next Steps ---");
        console.log("Use the above posterior expected returns and (posterior or prior) covariance matrix for portfolio optimization.");

    } catch (error) {
        if (error instanceof Error) {
            console.error("Error occurred during Black-Litterman model calculation:", error.message);
        } else {
            console.error("Unknown error occurred during Black-Litterman model calculation:", error);
        }
    }
}

// Run usage example
exampleUsage();

/*
Execution instructions:
1. Save this code as a file named `blackLitterman.ts`.
2. Make sure Node.js and TypeScript are installed.
   `npm install -g typescript ts-node`
3. Install mathjs.
   `npm install mathjs @types/mathjs`
4. Run the following in your terminal:
   `node --loader ts-node/esm blackLitterman.ts`
*/
