<img src="/img/geo_data.png" width=1200>

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

Columns to use in <i>market_sectors__historical_market</i>:

* age_median
* airport_volume (Total airport passenger volume. Measured at the zip code level based on the nearest airport)
* asset_value_momentum (Compares the year-over-year and trailing-twelve month change in asset values)
* desirability_quintile (Measures how desirable a market is to live in)
  - Somewhat Desirable (5)
  - Desirable (4)
  - Somewhat Desirable (3)
  - Less Desirable (2)
  - Much Less Desirable (1)

* fiscal_health_tax_quintile (Measures the financial viability and solvency of a market)
  - Healthy (3)
  - Stable (2)
  - Concerning (1)
* interstate_distance
* interstate_miles (The total miles of interstate with in a market)
* mrevpaf_growth_yoy_credit (The year-over-year growth in M-RevPAF, which combines two key operating metrics (rent and occupancy) into a single value) (75% full)
* occupancy (Percentage of total unit count that is physically occupied)
* population_500mi 


## ML notes

We can try to predict some financial metric based on its prediction & data from the past 6mo
<div style="color:yellow;"><b>BEWARE OF AUTOCORRELATION<b></div>

Candidates for y-values :
* ncf_growth
* noi_growth
* rent_growth
