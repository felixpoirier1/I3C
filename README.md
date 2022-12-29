# Green street notes

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

<img src="/img/geo_data.png" width=200>
