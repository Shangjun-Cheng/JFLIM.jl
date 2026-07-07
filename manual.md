# TRAnisotropy_global.jl Manual

## 1. Purpose of the script

`examples/TRAnisotropy_global.jl` performs a two-channel, global biexponential FLIM-style decay fit on time-resolved fluorescence data stored in a CSV file. The file name and column names indicate a time-resolved anisotropy dataset, but the current script does not explicitly compute an anisotropy curve or fit a rotational anisotropy model.

The practical purpose of the script is to:

- load two decay traces, `ch1_decay` and `ch2_decay`;
- crop both traces to a manually selected time-bin window;
- place the two channels into one fitting array, `to_fit`;
- fit both channels with a shared two-exponential lifetime model using `JFLIM.flim_fit`;
- compare measured data and fitted forward curves in a two-panel plot.

The global aspect is implemented by sharing lifetime parameters across the two channels while allowing channel-dependent amplitudes. This corresponds to a global decay analysis, not a full time-resolved anisotropy analysis with explicit parallel/perpendicular intensity equations.

## 2. Input data and file structure

The script activates the local Julia project:

```julia
Pkg.activate(raw"D:\JuliaProgramm\JFLIM")
Pkg.instantiate()
```

The input CSV path is hard-coded:

```julia
csv_path = raw"C:\Users\CHENG\Desktop\PaperSingleFiber\ref\5_decays_anisotropy.csv"
```

The script reads the file with:

```julia
df = CSV.read(csv_path, DataFrame)
```

Only two columns are used:

- `ch1_decay`
- `ch2_decay`

The repository also contains an example CSV at `examples/data/5_decays_anisotropy.csv`, whose header includes `time_ns`, `ch1_decay`, `ch2_decay`, `ch1_fit`, `ch2_fit`, `anisotropy`, and `anisotropy_from_fits`. However, the current script does not reference this repository-local file. It uses the absolute `csv_path` above.

The `time_ns`, `anisotropy`, and `anisotropy_from_fits` columns are not loaded or used by `TRAnisotropy_global.jl`. This step is not explicitly implemented in the current script.

## 3. Data loading workflow

The loading workflow is:

1. Read the CSV into a `DataFrame` named `df`.
2. Extract `df[!, "ch1_decay"]` into `y1`.
3. Extract `df[!, "ch2_decay"]` into `y2`.
4. Convert both vectors to `Float32`.
5. Define `T` as the minimum of the two channel lengths.
6. Truncate both vectors to length `T`.

The relevant variables are:

- `df`: table returned by `CSV.read`.
- `y1`: `Float32` vector holding `ch1_decay`.
- `y2`: `Float32` vector holding `ch2_decay`.
- `T`: common channel length after length matching.

No row filtering is done before the manual fit-window selection. In particular, the script comment says "NO zero truncation", and there is no search for the first nonzero sample.

## 4. Preprocessing and data preparation

The explicit preprocessing in the script is limited to length matching, manual windowing, data type conversion, and reshaping into a fitting array.

The fit window is manually specified in zero-based bin coordinates:

```julia
t_start = 391
t_end   = 796
```

The assertion

```julia
@assert 0 <= t_start <= t_end < T
```

checks that the selected window is valid. The script then converts the zero-based window to Julia's one-based index range:

```julia
idx = (t_start+1):(t_end+1)
Tfit = length(idx)
```

The fitting array is constructed as:

```julia
to_fit = Array{Float32}(undef, 2, 1, 1, Tfit)
to_fit[1,1,1,:] = y1[idx]
to_fit[2,1,1,:] = y2[idx]
```

Although the script comment describes a layout `(X, Y, Z, T, C) = (2, 1, 1, T, 1)`, the actual `to_fit` array is four-dimensional with layout `(X, Y, Z, T) = (2, 1, 1, Tfit)`. The two channels are stored along the first dimension, `X`.

The following common preprocessing steps are not implemented:

- Background subtraction: This step is not explicitly implemented in the current script. A fitted/model offset is used inside `flim_fit`, but the raw `to_fit` data are not background-subtracted before fitting.
- Normalization by total intensity or area: This step is not explicitly implemented in the current script. `ScaleMaximum()` scales the numerical fitting problem by the maximum count value inside `flim_fit`, then rescales the result.
- Time calibration from `time_ns`: This step is not explicitly implemented in the current script. The plot uses bin indices, not nanoseconds.
- ROI selection or mask handling: This step is not explicitly implemented in the current script.
- Smoothing or denoising: This step is not explicitly implemented in the current script.
- IRF loading or convolution: This step is not explicitly implemented in the current script because no `irf` argument is passed to `flim_fit`.

Inside `flim_fit`, a temporary `to_fit_no_offset = max.(to_fit .- off_start, 0f0)` is used for initial lifetime and amplitude estimation. This temporary array does not replace the measured `to_fit` used by the loss function.

## 5. Construction of anisotropy-related signals

The script treats the two CSV decay columns as two related channels:

- `ch1_decay` -> `to_fit[1,1,1,:]`
- `ch2_decay` -> `to_fit[2,1,1,:]`

The script does not explicitly label these channels as parallel or perpendicular. It also does not compute a standard anisotropy signal such as:

```text
r(t) = (I_parallel(t) - G * I_perpendicular(t)) /
       (I_parallel(t) + 2 * G * I_perpendicular(t))
```

No `G` factor is defined, fitted, or applied. The CSV columns `anisotropy` and `anisotropy_from_fits`, if present, are ignored. Therefore, the current script uses anisotropy-related input channels only as two decay datasets for a shared-lifetime global fit.

This means the fitted model is a two-channel intensity decay model, not an explicit anisotropy decay model.

## 6. Global fitting mechanism

The script performs two sequential calls to `flim_fit`.

### First fit

The first fit is:

```julia
res, fwd = flim_fit(
    to_fit, ScaleMaximum();
    use_cuda=true,
    verbose=true,
    stat=loss_gaussian,
    iterations=30,
    num_exponents=2,
    fixed_tau=true,
    global_tau=true,
    amp_positive=true,
    off_start=0.0001f0,
    bgnoise=1f0
)
```

Important behavior:

- `ScaleMaximum()` divides the data, amplitudes, and offset by `maximum(to_fit)` during fitting, then rescales fitted amplitudes, offset, and forward model.
- `num_exponents=2` selects a biexponential decay model.
- `fixed_tau=true` fixes the lifetime values during this first optimization.
- `global_tau=true` makes the starting lifetime estimate global by averaging over dimensions `(1, 2)` in `flim_fit`.
- `amp_positive=true` wraps amplitudes in `Positive`, forcing amplitudes to be non-negative through the `InverseModeling` parameter transform.
- `off_start=0.0001f0` provides the starting offset.
- `fixed_offset` is not passed, so it defaults to `true` in `flim_fit`; the offset is fixed in the first fit.
- `stat=loss_gaussian` uses a sum-of-squared-errors objective.

If `tau_start` is not supplied, `flim_fit` estimates an initial lifetime with `get_tau`, which uses a rapid-lifetime-determination style ratio of two integrated time windows. For `num_exponents=2`, the initial lifetime estimate is duplicated along the exponent dimension and scaled by factors `0.5` and `1.5`.

### Second fit

The second fit is:

```julia
res2, fwd2 = flim_fit(
    to_fit, ScaleMaximum();
    use_cuda=true,
    verbose=true,
    stat=loss_anscombe_pos,
    iterations=80,
    num_exponents=2,
    fixed_tau=false,
    global_tau=true,
    fixed_offset=false,
    amp_positive=true,
    all_start=res,
    bgnoise=1f0
)
```

Important behavior:

- `all_start=res` reuses the first fit as the initial parameter set.
- `fixed_tau=false` makes the lifetime parameters optimizable.
- `global_tau=true` remains active, and the reused lifetime array from `res` has the global shape produced by the first fit; therefore the two channels share the optimized lifetimes.
- `fixed_offset=false` makes the offset optimizable.
- `stat=loss_anscombe_pos` uses the positive Anscombe loss with `bgnoise=1f0`; negative data and predictions are clipped to non-negative values inside this loss function.
- `iterations=80` sets the maximum optimizer iterations for this stage.

### Global and local parameters

In this script:

| Parameter | Global or local | Notes |
|---|---|---|
| `τs` / lifetime components | Global | Shared across the two channels because `global_tau=true` produces a singleton channel dimension for lifetimes. |
| `amps` | Local | Broadcast over channel dimension; each channel has its own amplitudes for each exponential component. |
| `offset` | Global scalar-like parameter | The script supplies a single `off_start`; in the second fit this becomes one positive fitted offset shared by the model. |
| `t0` | Not used | No `irf` is passed, so `t0_start` remains `nothing` and no temporal shift is fitted. |

The forward model is constructed in `JFLIM.get_fwd`. Since `irf=nothing`, it returns a closure around `multi_exp`. The active model is:

```text
fit(channel, time) = sum over exponent components of
                     offset + amp[channel, component] * exp(-time / tau[component])
```

This description follows the implementation of `multi_exp`, which calls:

```julia
sum(offset .+ amps .* exp.(-time_data ./ τs), dims=5)
```

Because `offset` is inside the sum over exponent components, the effective constant contribution is repeated once per exponential component. With `num_exponents=2`, the implemented expression contributes `2 * offset` to the fitted curve if `offset` is scalar-like.

The objective function is created by `InverseModeling.loss(to_fit, forward, stat, bgnoise)`. The optimizer is `InverseModeling.optimize_model`, which uses `Optim.optimize` with the default `LBFGS()` optimizer and gradients computed through `Zygote.pullback`.

## 7. Model interpretation

The active physical/empirical model is a biexponential fluorescence intensity decay in time-bin units:

```text
I_c(t) = sum_k [ b + A_c,k * exp(-t / tau_k) ]
```

where:

- `c` is the channel index stored in the first array dimension;
- `k` is the exponential component index;
- `A_c,k` is the channel-specific amplitude;
- `tau_k` is the globally shared lifetime;
- `b` is the fitted or fixed offset term;
- `t` is the bin index, not calibrated physical time.

The fitted lifetimes are therefore in units of bins. The script does not multiply by a time step and does not use the CSV `time_ns` column.

Rotational correlation times, initial anisotropy `r0`, anisotropy decay components, and `G`-factor correction are not active model parameters. This step is not explicitly implemented in the current script.

`JFLIM.flim_fit` contains support for an IRF path through `multi_exp_irf`, `plan_conv_psf`, and `conv_aux`, but `TRAnisotropy_global.jl` does not pass an `irf` argument. Therefore, IRF convolution is not used in this run. This step is not explicitly implemented in the current script.

## 8. Output results

The script produces in-memory fit results and a displayed plot. It does not save a results table, parameter file, or image by default.

In-memory outputs:

- `res`: parameter result from the first fit.
- `fwd`: forward-model fit curve from the first fit.
- `res2`: parameter result from the second fit.
- `fwd2`: forward-model fit curve from the second fit.
- `amps = res2[:amps]`: final fitted amplitudes.
- `τs = res2[:τs]`: final fitted lifetimes.
- `res2.offset`: final fitted offset.

Console output:

- `Pkg.activate` and package messages may appear.
- `@show size(to_fit), maximum(y1), maximum(y2)` prints the fitting array size and channel maxima.
- `flim_fit(..., ScaleMaximum())` prints scaling information.
- With `verbose=true`, `flim_fit` prints initial loss values, timing information, and optimizer trace information from `Optim`.
- The final parameter expressions `res[:τs]`, `τs = res2[:τs]`, and `res2.offset` are not explicitly printed with `println` or `@show` inside the function.

Plot output:

- `p1`: channel 1 data and fitted curve.
- `p2`: channel 2 data and fitted curve.
- `p`: a two-row layout containing `p1` and `p2`.
- `display(p)` shows the plot.

The x-axis is `t = collect(0:(Tfit-1))`, so it is a window-local bin index. It is not the original CSV row index and not physical time in nanoseconds.

Residuals are not computed or plotted in the current script. This step is not explicitly implemented in the current script.

File output:

- No file is saved by default.
- The line `savefig(p, "to_fit_vs_fwd2.png")` is present only as a comment.

## 9. Key variables and functions

| Name | Type | Role |
|---|---|---|
| `csv_path` | string | Hard-coded absolute path to the input CSV file. |
| `df` | `DataFrame` | Table loaded from the CSV file by `CSV.read`. |
| `y1` | `Vector{Float32}` | Decay data from `ch1_decay`. |
| `y2` | `Vector{Float32}` | Decay data from `ch2_decay`. |
| `T` | integer | Common length of `y1` and `y2` after truncation to the shorter channel. |
| `t_start`, `t_end` | integers | Manual zero-based inclusive fit-window bounds. |
| `idx` | range | One-based Julia index range corresponding to the selected fit window. |
| `Tfit` | integer | Number of time bins in the fit window. |
| `to_fit` | `Array{Float32,4}` | Fitting data array with actual layout `(2, 1, 1, Tfit)`. |
| `use_cuda` | `Bool` | Enables GPU fitting through `CuArray` conversion inside `flim_fit`. |
| `amp_positive` | `Bool` | Requests non-negative amplitude constraints. |
| `res`, `fwd` | named tuple / array | First-stage fixed-lifetime Gaussian fit results and fitted curve. |
| `res2`, `fwd2` | named tuple / array | Second-stage free-lifetime positive-Anscombe fit results and fitted curve. |
| `amps` | array | Final fitted amplitudes extracted from `res2[:amps]`. |
| `τs` | array | Final fitted lifetimes extracted from `res2[:τs]`. |
| `t` | vector | Window-local bin axis for plotting. |
| `data_ch1`, `data_ch2` | vectors | Measured channel traces extracted from `to_fit`. |
| `fit_ch1`, `fit_ch2` | vectors | Fitted channel traces extracted from `fwd2`. |
| `p1`, `p2`, `p` | plot objects | Channel-level plots and combined two-panel figure. |
| `CSV.read` | function | Loads CSV data into a `DataFrame`. |
| `flim_fit` | function from `JFLIM` | Main fitting wrapper for FLIM multi-exponential decay models. |
| `ScaleMaximum` | scaling type from `JFLIM` | Scales data by the maximum value during fitting. |
| `get_tau` | function from `JFLIM` | Estimates initial lifetime values using integrated decay-window ratios. |
| `multi_exp` | function from `JFLIM` | Active no-IRF multi-exponential forward model. |
| `get_start_vals` | function from `JFLIM` | Wraps parameter starts as `Fixed`, `Positive`, or raw values. |
| `create_forward` | function from `InverseModeling` | Splits fixed and fitted parameters and builds the differentiable forward model. |
| `loss_gaussian` | function from `InverseModeling` | First-stage sum-of-squared-error loss. |
| `loss_anscombe_pos` | function from `InverseModeling` | Second-stage positive Anscombe loss. |
| `optimize_model` | function from `InverseModeling` | Runs LBFGS optimization with automatic differentiation. |
| `plot`, `plot!`, `display` | functions from `Plots` | Build and display the final data-vs-fit figure. |

## 10. Complete data-processing pipeline

```text
Hard-coded CSV path
-> CSV.read(..., DataFrame)
-> extract ch1_decay and ch2_decay
-> convert to Float32
-> truncate both channels to common length T
-> manually select zero-based fit window 391:796
-> convert to Julia one-based idx
-> build to_fit with channels in X dimension
-> call flim_fit stage 1:
   ScaleMaximum, two exponentials, fixed global lifetimes, fixed offset,
   positive amplitudes, Gaussian loss
-> call flim_fit stage 2:
   reuse stage-1 result, free positive global lifetimes,
   free positive offset, positive amplitudes, positive Anscombe loss
-> extract amplitudes, lifetimes, and offset from res2
-> extract fitted channel curves from fwd2
-> plot measured channel traces and fitted curves against bin index
-> display plot
```

## 11. Notes and possible limitations

- The input path is hard-coded to an absolute user-specific location. The repository-local CSV under `examples/data/` is not used by this script.
- The script comments describe a five-dimensional `(X, Y, Z, T, C)` layout, but the implemented `to_fit` array is four-dimensional.
- `use_cuda=true` requires a working CUDA setup. If CUDA is unavailable, the script may fail unless `use_cuda` is changed.
- The script does not use the CSV time column. All fitted lifetimes and plot x-values are in bin units.
- No explicit parallel/perpendicular convention is enforced. The meaning of `ch1_decay` and `ch2_decay` must be known from the data source.
- No anisotropy curve is calculated. This step is not explicitly implemented in the current script.
- No `G`-factor correction is applied. This step is not explicitly implemented in the current script.
- No rotational correlation-time model is fitted. This step is not explicitly implemented in the current script.
- No IRF is supplied to `flim_fit`; therefore, IRF convolution is not used. This step is not explicitly implemented in the current script.
- Background is modeled through an offset parameter but is not subtracted from the raw data before fitting.
- The implemented `multi_exp` expression places `offset` inside the sum over exponential components, so the effective constant baseline is repeated for each component.
- `num_exponents=2` is fixed in both fitting stages. Single-component or higher-order models are not explored.
- Residuals are not computed, plotted, or saved. This step is not explicitly implemented in the current script.
- No output figure or parameter table is saved unless the commented `savefig` line is manually enabled.
