

/* 1. CREAREA UNUI SET DE DATE SAS DIN FIȘIERE EXTERNE */
proc import datafile="/home/u64207861/US_Accidents_Sample_1000_Per_Year.csv"
    out=work.accidents
    dbms=csv
    replace;
    guessingrows=max;
run;

/* 2. CREAREA ȘI FOLOSIREA DE FORMATE DEFINITE DE UTILIZATOR */
proc format;
  value severity_fmt
    1 = "Minor"
    2 = "Moderate"
    3 = "Serious"
    4 = "Severe";
    
  value $weather_fmt
    'Clear' = 'Good Conditions'
    'Cloudy', 'Overcast' = 'Moderate Conditions'
    'Rain', 'Snow', 'Fog' = 'Poor Conditions'
    other = 'Variable Conditions';
    
  value distance_fmt
    low-0.5 = 'Very Short'
    0.5<-1 = 'Short'
    1<-3 = 'Medium'
    3<-high = 'Long';
run;

data accidents_fmt;
  set accidents;
  format Severity severity_fmt. 
         Weather_Condition $weather_fmt.
         'Distance(mi)'n distance_fmt.;
run;

/* 3. PROCESAREA ITERATIVĂ ȘI CONDIȚIONALĂ A DATELOR */
data accidents_dur;
  set accidents;
  format Start_Time End_Time datetime.;
  Duration = intck('minute', Start_Time, End_Time);

  if Duration < 10 then DurCategory = "Short";
  else if Duration < 60 then DurCategory = "Medium";
  else if Duration >= 60 then DurCategory = "Long";
  
  /* Procesare condițională pentru crearea categoriilor de risc */
  if Severity >= 3 and 'Distance(mi)'n > 1 then Risk_Level = "High";
  else if Severity >= 2 or 'Distance(mi)'n > 0.5 then Risk_Level = "Medium";
  else Risk_Level = "Low";
run;

/* 4. CREAREA DE SUBSETURI DE DATE */
data severe_accidents;
  set accidents;
  where Severity >= 3 and 'Distance(mi)'n > 1;
run;

data accidents_by_state;
  set accidents;
  where State in ('CA', 'TX', 'FL', 'NY');
run;

data night_accidents;
  set accidents;
  where 6 <= hour(Start_Time) <= 18;
  keep ID State Severity Start_Time Weather_Condition;
run;

/* 5. UTILIZAREA DE FUNCȚII SAS */
data accidents_time;
  set accidents;
  Year  = year(datepart(Start_Time));
  Month = month(datepart(Start_Time));
  Hour  = hour(Start_Time);
  Day_of_Week = weekday(datepart(Start_Time));
  
  /* Funcții de manipulare text */
  State_Upper = upcase(State);
  City_Length = length(City);
  
  /* Funcții matematice */
  Temp_Celsius = ('Temperature(F)'n - 32) * 5/9;
  Log_Distance = log('Distance(mi)'n + 1);
run;

/* 6. UTILIZAREA DE MASIVE */
data weather_avg;
  set accidents;
  array weather[3] 'Temperature(F)'n 'Humidity(%)'n 'Pressure(in)'n;
  array weather_norm[3] temp_norm humid_norm press_norm;
  
  total = 0;
  do i = 1 to dim(weather);
    if not missing(weather[i]) then total + weather[i];
  end;
  avg_weather = total / dim(weather);
  
  /* Normalizare folosind masive */
  do i = 1 to dim(weather);
    if not missing(weather[i]) then 
      weather_norm[i] = (weather[i] - avg_weather) / avg_weather;
  end;
  
  drop i total;
run;

/* 7. COMBINAREA SETURILOR DE DATE PRIN SQL */
proc sql;
  create table accidents_summary as
  select State,
         count(*) as Total_Accidents,
         avg(Severity) as Avg_Severity,
         avg('Distance(mi)'n) as Avg_Distance,
         max('Temperature(F)'n) as Max_Temp,
         min('Temperature(F)'n) as Min_Temp
  from accidents
  group by State
  having count(*) > 10
  order by Total_Accidents desc;
quit;

/* Combinare prin INNER JOIN */
proc sql;
  create table state_weather_analysis as
  select a.State,
         a.Weather_Condition,
         count(*) as Accident_Count,
         s.Total_Accidents,
         (calculated Accident_Count / s.Total_Accidents * 100) as Percentage
  from accidents a
  inner join accidents_summary s
  on a.State = s.State
  group by a.State, a.Weather_Condition, s.Total_Accidents
  order by a.State, calculated Percentage desc;
quit;

/* 8. UTILIZAREA DE PROCEDURI PENTRU RAPORTARE */
proc report data=accidents_dur nowd;
  column State Severity Duration ('Weather Stats' 'Temperature(F)'n 'Humidity(%)'n);
  define State / group "State";
  define Severity / group "Severity Level";
  define Duration / analysis mean "Avg Duration (min)" format=8.2;
  define 'Temperature(F)'n / analysis mean "Avg Temp (°F)" format=8.1;
  define 'Humidity(%)'n / analysis mean "Avg Humidity (%)" format=8.1;
  
  where State in ('CA', 'TX', 'FL', 'NY', 'OH');
  title "Accident Analysis by State and Severity";
run;

/* Raport cu opțiuni avansate */
proc tabulate data=accidents_dur;
  class State Severity DurCategory;
  var Duration 'Distance(mi)'n;
  table State,
        Severity * (DurCategory * (Duration * mean 'Distance(mi)'n * mean));
  title "Cross-tabulation Analysis of Accidents";
run;

/* 9. FOLOSIREA DE PROCEDURI STATISTICE */
proc means data=accidents_dur n mean median std min max;
  var Duration 'Distance(mi)'n 'Temperature(F)'n 'Humidity(%)'n;
  class State;
  title "Descriptive Statistics by State";
run;

proc freq data=accidents;
  tables Severity Weather_Condition / chisq;
  tables State * Severity / chisq expected;
  title "Frequency Analysis and Chi-Square Tests";
run;

/* Analiză de corelație */
proc corr data=accidents pearson spearman;
  var Severity 'Distance(mi)'n 'Temperature(F)'n 'Humidity(%)'n 'Pressure(in)'n;
  title "Correlation Analysis of Numeric Variables";
run;

/* Test t pentru compararea temperaturilor între severități */
proc ttest data=accidents;
  class Severity;
  var 'Temperature(F)'n;
  where Severity in (1, 4);
  title "T-Test: Temperature Comparison between Minor and Severe Accidents";
run;

/* 10. GENERAREA DE GRAFICE */
proc sgplot data=accidents_dur;
  histogram Duration;
  density Duration;
  title "Distribution of Accident Duration";
run;

proc sgplot data=accidents_dur;
  vbox Duration / category=Severity;
  title "Duration Distribution by Severity Level";
run;

/* Grafice avansate */
proc sgplot data=accidents;
  scatter x='Temperature(F)'n y='Humidity(%)'n / group=Severity;
  title "Temperature vs Humidity by Severity";
run;

proc sgplot data=accidents_summary;
  hbar State / response=Total_Accidents datalabel;
  title "Total Accidents by State";
run;

/* Heatmap pentru analiza temporală */
data time_analysis;
  set accidents;
  Hour = hour(Start_Time);
  Day_of_Week = weekday(datepart(Start_Time));
  if not missing(Hour) and not missing(Day_of_Week);
run;

proc freq data=time_analysis noprint;
  tables Day_of_Week * Hour / out=freq_time;
run;

proc sgplot data=freq_time;
  heatmap x=Hour y=Day_of_Week / colorresponse=Count;
  title "Accident Frequency Heatmap: Hour vs Day of Week";
run;

/* 11. ANALIZĂ STATISTICĂ AVANSATĂ */
/* Analiză de regresie multiplă */
proc reg data=accidents;
  model Severity = 'Temperature(F)'n 'Humidity(%)'n 'Pressure(in)'n 'Distance(mi)'n;
  title "Multiple Regression: Factors Affecting Accident Severity";
run;

/* Analiză de varianță (ANOVA) */
proc anova data=accidents;
  class Weather_Condition;
  model Severity = Weather_Condition;
  means Weather_Condition / tukey;
  title "ANOVA: Severity by Weather Condition";
run;

/* Analiză non-parametrică */
proc npar1way data=accidents wilcoxon;
  class Weather_Condition;
  var Severity;
  title "Non-parametric Test: Severity by Weather Condition";
run;

/* 12. MACHINE LEARNING ÎN SAS */
/* Pregătirea datelor pentru ML */
data ml_prep;
  set accidents;
  where not missing(Severity) and not missing('Temperature(F)'n) and 
        not missing('Humidity(%)'n) and not missing('Distance(mi)'n);
  
  /* Encoding variabile categoriale */
  if Weather_Condition = 'Clear' then Weather_Clear = 1; else Weather_Clear = 0;
  if Weather_Condition = 'Rain' then Weather_Rain = 1; else Weather_Rain = 0;
  if Weather_Condition = 'Cloudy' then Weather_Cloudy = 1; else Weather_Cloudy = 0;
  
  /* Crearea variabilei target binare */
  if Severity >= 3 then High_Severity = 1; else High_Severity = 0;
run;

/* Împărțirea în train și test */
proc surveyselect data=ml_prep out=ml_split outall
  method=srs rate=0.7 seed=12345;
run;

data ml_train ml_test;
  set ml_split;
  if selected then output ml_train;
  else output ml_test;
run;

/* Model de regresie logistică */
proc logistic data=ml_train descending;
  model High_Severity = 'Temperature(F)'n 'Humidity(%)'n 'Distance(mi)'n 
                       Weather_Clear Weather_Rain Weather_Cloudy;
  score data=ml_test out=predictions;
  title "Logistic Regression: Predicting High Severity Accidents";
run;

/* Evaluarea modelului */
data model_eval;
  set predictions;
  predicted_class = (p_1 > 0.5);
run;

proc freq data=model_eval;
  tables High_Severity * predicted_class / agree;
  title "Model Performance: Confusion Matrix";
run;

/* Random Forest cu PROC HPFOREST */
proc hpforest data=ml_train seed=123 maxtrees=100;
  target High_Severity / level=binary;
  input 'Temperature(F)'n 'Humidity(%)'n 'Distance(mi)'n 
        Weather_Clear Weather_Rain Weather_Cloudy;
  score out=rf_scored;
  title "Random Forest Model for Accident Severity Prediction";
run;

/* BONUS: TRANSFORMĂRI AVANSATE DE DATE */
/* Transpunerea datelor pentru analiză temporală */
proc sort data=accidents_summary;
  by State;
run;

proc transpose data=accidents_summary out=state_transposed;
  by State;
  var Total_Accidents Avg_Severity Avg_Distance;
run;

/* Crearea de variabile lag pentru analiză de serie temporală */
proc sort data=accidents;
  by State Start_Time;
run;

data time_series;
  set accidents;
  by State;
  
  lag_severity = lag(Severity);
  if first.State then lag_severity = .;
  
  severity_change = Severity - lag_severity;
run;

/* RAPORT FINAL */
proc print data=accidents_summary (obs=10);
  title "Top 10 States by Accident Frequency - Final Summary";
run;

/* Exportul rezultatelor */
proc export data=accidents_summary
  outfile="/home/u64207861/accidents_summary.csv"
  dbms=csv replace;
run;

/* Salvarea formatelor permanente */
proc format library=work.formats;
  value severity_fmt
    1 = "Minor"
    2 = "Moderate" 
    3 = "Serious"
    4 = "Severe";
run;

title; 