<img src="/img/geo_data.png" width=1200>
<img src="/img/NCF_Top10.png" width=2000>

# Assumptions
Here are the assumptions underlying the final model, which will try to find price inefficiencies:
* Relative to "forecast" data (y-val to be predicted):
	1. GS baseline forecasts are most likely to materialize given a large enough time interval
	2. CF forecasts follow market expectations are unlikely to deviate by a large amount

* Relative to "dependant" data (y-val to be trained/tested/cross-validated):
	1. Data compiled by GS is accurate and represents a regional RE market holistically
	2. Methods of measurment are constant throughout time

* Relative to "regional" data (x_itn data):
	1. Regional data is unlikely to experience large changes through time
	2. Regional data are mean reverting, meaning they subscribe to no directional trend

* Relative to "macro" data (x_tn data):
	1. The expected effect of macro data is equal across markets
	2. Macro data has explanatory value regarding fluctuations in RE valuation

* Relative to "geographic" data (x_lln data):
	1. Geographic data is unlikely to experience large changes through time
	2. Geographic data is constant throughout the vector of time (subject to change*)

*if there is enough time, we could simulate monte-carlo simulation, by perverting the geographic landscape of a region and seeing how it affects its RE potential

# Notes
## Green street notes

<table>
  <tr>
    <th>file</th>
    <th>desc</th>
    <th>use</th>
  </tr>
  <tr>
    <td>forecasts__historical_baseline</td>
    <td>longitudinal of all cities form 2021-Q1 to 2022-Q1 with net cash flow growth and net operating income growth forecasts until 2027</td>
    <td>great source to extract overall performance by city, industrial contains 15.5k obs of 51 markets</td>
  </tr>
  <tr>
    <td>forecasts__historical_exceptionally_strong_growth</td>
    <td>same markets, but only including properties who've performed well</td>
    <td>Same data as last but with different forecast expectation</td>
  </tr>
    <td> ... </td>
	<td> ... </td>
	<td> ... </td>
  </tr>
  <tr>
    <td>forecasts_scenarios_baseline</td>
	<td>Same as previous but only fc published on 2022-Q1 (latest fc)</td>
	<td>Saves time but can simply query other database, don't waste your time</td>
  </tr>
  <tr>
    <td>market_companies__summaries_market_na</td>
	<td>8 companies studied by region with interesting metric</td>
	<td>may be interesting later, as companies are labelled by region, but likely overkill</td>
  </tr>
  <tr>
    <td>market_sectors__historical_market</td>
	<td>statistics with all markets observed in other datasets, contains lon lat !!!</td>
	<td>goldmine, can be used in geographic model to have more insight</td>
  </tr>
  <tr>
    <td>market_sectors__historical_submarket</td>
	<td>statists with markets further granulated, still has lon lat!!! </td>
	<td> another goldmine</td>
  </tr>


</table>

# Data selection and justification

### Regional data x_itn (max 15)
Columns to use in <i>market_sectors__historical_market</i> <b> ("regional" data x_itn) </b>:

* age_median (we assume age to be an important indocator of a population's propensity to buy goods and stimulate the local economy)
* airport_volume (Total airport passenger volume. Measured at the zip code level based on the nearest airport) (since airports locations are already included in geographic model and that it somewhat takes into consideration volume, this may be trivial)
* asset_value_momentum (Compares the year-over-year and trailing-twelve month change in asset values) (Momentum is generally a good sign of futur growth, still unsure if it should be added)
* desirability_quintile (Measures how desirable a market is to live in) (somewhat subjective and its measurement is unclear, but if accurate could be very useful)
  - Very Desirable (5)
  - Desirable (4)
  - Somewhat Desirable (3)
  - Less Desirable (2)
  - Much Less Desirable (1)

* fiscal_health_tax_quintile (Measures the financial viability and solvency of a market) (very useful, credit risk has to be inserted somewhere and local credit risk is optimal, specifically when included with a macro indicator)
  - Healthy (3)
  - Stable (2)
  - Concerning (1)
* interstate_distance (interstates could not be added in the geographic model due to complex data structures, this is a shortcut, not a feature)
* interstate_miles (The total miles of interstate with in a market) (still a copout, but slightly better)
* mrevpaf_growth_yoy_credit (The year-over-year growth in M-RevPAF, which combines two key operating metrics (rent and occupancy) into a single value) (75% full) (quite black-box, but if accurate very useful to describe the actual utilization rate of properties)
* occupancy (Percentage of total unit count that is physically occupied) (perhaps this metric should trump previous metric)
* population_500mi (useful, quite broad, may be able to somewhat complement the urban center geo map)

Other potential <b> ("regional" data x_itn) </b>:

### Macro data (max 5)
* Interest rates (obviously preferably for same maturity as the avg logistic center mortgage)
* Exchange rates (especially USDMEX & USDCAD & EURUSD) (NAFTA is a huge part of the US's exports, this matters quite a lot)
* PMI (Purchasing Manager's Index is a great forward looking indicator that incorporates business confidence, which are generally the owners of logistics centers)
* WTICO
* EX - IM (trade balance)

## ML notes

We can try to predict some financial metric based on its prediction & data from the past 6mo

<b>BEWARE OF AUTOCORRELATION</b>

Candidates for y-values :
* ncf_growth
* noi_growth
* rent_growth
