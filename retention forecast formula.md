## B (Adj):

"if there's no error, return the first argument. If there's an error, return the second (blank)"
=iferror( 

    "if logical expression is true, return 1st argument, otherwise return 2nd.
    "if the cohort date is less than the forecast date, then return a blank, otherwise forecast.
    if(

        "DATEDIF returns the count of months between the two dates."
        DATEDIF($B13,$B$4,"m")<4, 
        
        "", 

        "Return the content of a cell specified by the row and column. First arg is cell range to
        look at, 2nd arg is row, and 3rd is column. In this case, INDEX is used to just get the slope
        and not the intercept from the LINEST function."
        INDEX(

            "LINEST(known_data_y, [known_data_x], [calculate_b], [verbose])"
            LINEST(

                "natural log"
                LN( 
                    "First index returns the 1st value of the cohort (100%). Second index 
                    returns the last value of the cohort. The colon in between creates an array.
                    So this basically returns the array of values for the given cohort"
                    INDEX($BD13:$CA13,1,1): INDEX($BD13:$BQ13,1,DATEDIF($B13,$B$4,"m")-1)
                    ), 
                LN( 
                    "Array of values for the pLTV curve"
                    INDEX($BD$6:$BQ$6,1,1): INDEX($BD$6:$CA$6,1, DATEDIF($B13,$B$4,"m")-1)
                    ),,
                )
            , 1)

        "Now we take the computed slope from LINEST, add 2% * (0 | 6 - Final month)
        ) + 2% * MAX(0,6 - (DATEDIF($B13,$B$4,"m")-1)) 
    
    ,"")


## Retention Forecast

=if(
    "If diff between cohort date and current month is larger than t. In other words,
    if we have actuals, return the actuals, else compute the forecast."
    DATEDIF($B44,$B$4,"m")>BD$6,
    BD13,

    "previous value * (max monthly borrower | (A * t**B)/(A * (t-1)**B)) Whichever is smaller.
    BC44*MIN(Inputs!$D$12,($BB44*BD$6^$BC44)/($BB44*BC$6^$BC44))
    
)