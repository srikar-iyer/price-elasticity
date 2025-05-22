## Average Price Elasticity Calculation

### Scripts

To generate the output file with price elasticity, if given an executable, execute the following command in Command Prompt

.\price_elasticity.exe input.csv -o output.csv 

Or, if given a python file, cd ./file/path and execute python -m price_elasticity (or whatever the python file name is) input.csv -o output.csv

Then, to run python script as a very heavy exe, we can run : pyinstaller --onefile (name of python file), then cd .\dist\, then run the exe command above.

### Input/Output Excel/csv Schema

Data in the input.csv should be in the following format.

Store_Id,Item,proc_date,Category_Name,Unit_Retail,Total_units
104,1800012575,1/2/2023,Frozen Bread,4.69,2
104,1800012575,1/8/2023,Frozen Bread,4.69,1

The output depends on what the code does; in our case, for the standard avg elasticity calculation, it looks like 

Item,Unit_Retail,Unit_Retail,Unit_Retail,Unit_Retail,Unit_Retail,Total_units,Total_units,Total_units,Total_units,Total_units,Item,Slope,Intercept,R_squared,predicted_min_quantity,predicted_max_quantity,predicted_median_quantity,predicted_average_quantity,min_avg_elasticity,max_avg_elasticity,avg_elasticity
,mean,sum,min,max,median,mean,sum,min,max,median,count,,,,,,,,,,
1000,1.9899998416695959,1132.30990991,1.98990991,1.99,1.99,18.36731107205624,10451.0,2.0,111.0,16.0,569,-1030034.1286251924,2049786.1201894851,0.10976754772388679,111.0,18.204225352266803,18.204225352266803,18.36731107207015,-31688.39824443022,-112096.34783365998,-71892.3730390451

and for the advanced elasticity calculation, it looks like : 

Unit_Retail,Unit_Retail,Unit_Retail,Unit_Retail,Unit_Retail,Unit_Retail,Unit_Retail,Total_units,Total_units,Total_units,Total_units,Total_units,Total_units,Total_units,Item,Slope,Intercept,R_squared,predicted_min_quantity,predicted_max_quantity,predicted_median_quantity,predicted_average_quantity,predicted_25pct_quantity,predicted_75pct_quantity,min_25pct_elasticity,25pct_median_elasticity,median_75pct_elasticity,75pct_max_elasticity,min_median_elasticity,median_max_elasticity,min_average_elasticity,average_max_elasticity,avg_quartile_elasticity,weighted_avg_quartile_elasticity,median_avg_elasticity,mean_avg_elasticity,valid_elasticity
mean,sum,min,max,median,25pcentile,75pcentile,mean,sum,min,max,median,25pcentile,75pcentile,count,,,,,,,,,,,,,,,,,,,,,,
1.9899998416695959,1132.30990991,1.98990991,1.99,1.99,1.99,1.99,18.36731107205624,10451.0,2.0,111.0,16.0,10.0,23.0,569,-1030034.1286251924,2049786.1201894851,0.10976754772388679,111.0,18.204225352266803,18.204225352266803,18.36731107207015,18.204225352266803,18.204225352266803,-31728.397619891606,,,,-31728.397619891606,,-31688.39824443022,-112096.34783365998,-31728.397619891606,,,-71892.3730390451,True

Input and Output files can have names other than given above. Above is an example.

### Advanced Elasticity Calculation Results


