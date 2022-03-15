"if there's no error, return the first argument. If there's an error, return the second (blank)"
=iferror( 

    "if logical expression is true, return 1st argument, otherwise return 2nd.
    "if the cohort date is less than the forecast date, then return a blank, otherwise forecast.
    if(

        "DATEDIF returns the count of months between the two dates."
        DATEDIF($B13,$B$4,"m")<4, 
        
        "", 

        "Return the content of a cell specified by the row and column. First arg is cell range to
        look at, 2nd arg is row, and 3rd is column."
        INDEX(

            "LINEST(known_data_y, [known_data_x], [calculate_b], [verbose])"
            LINEST(

                "natural log"
                LN( 
                    INDEX($BD13:$CA13,1,1): INDEX($BD13:$BQ13, 1, DATEDIF($B13,$B$4,"m")-1)
                    ), 
                LN( 
                    INDEX($BD$6:$BQ$6,1,1): INDEX($BD$6:$CA$6,1, DATEDIF($B13,$B$4,"m")-1)
                    ),,
                )
            , 1)

        ) + 2% * MAX(0,6 - (DATEDIF($B13,$B$4,"m")-1)) 
    
    ,"")