# Teknisk rapport: Analyse av straumforbruk og vêrdata i London

Av Terje Sørbø og Ørjan Tornvik
Gruppenavn: Terje og Ørjan si gruppe

## 1. Introduksjon

Denne rapporten byggjer på analysen i `delivery03-exam-london/analysis.ipynb`, der målet er å forstå korleis vêret påverkar straumforbruket til éin hushaldning i London (`MAC000002`), og å evaluere enkle tidsseriemodellar for korttidsprognosar.

### Problemstilling

Analysen tek føre seg fire hovudspørsmål:

1. Kva samanhengar finst mellom vêrvariablar og energiforbruk?
2. Kva tidsserie-features (laggar, glidande snitt, tid-på-døgnet) er nyttige?
3. Korleis oppfører serien seg med omsyn til stasjonaritet og sesongmønster?
4. Korleis presterer AR-, MA-, ARMA- og ARIMA-modellar for 48-timars prognosehorisont?

### Bruksområde

Bruksområdet er korttidsprognosering av straumforbruk for ein privat bustad. Dette er relevant for:

- Planlegging av last og energibruk i smarte nett
- Betre styring av oppvarming
- Operasjonelle prognosar med timeoppløysing

### Datasett

Tre datakjelder er fletta saman:

- **Energi (halvtimesdata):** `MAC000002_energy_halfhourly.csv` (okt 2012-feb 2014)
- **Vêr (timeoppløysing):** `London_weather_hourly.csv` (temperatur, vind, luftfukt, trykk)
- **Vêr (dagoppløysing):** `london_weather.csv` (soltimar, nedbør, skydekke, snødjupn)

Halvtimes energidata er aggregerte til timesnivå, deretter left-joined med vêrdata. Manglande verdiar er handterte med fjerning av nullrad i energidata, forward fill for enkelte vêrfelt, og dropping av rader utan full dekning etter samanslåing.

## 2. Resultat

### Data og feature engineering

Analysen etablerer eit samansett timesdatasett med energi + vêr, og utleiar fleire forklarande variablar:

- Kalenderfeature: time, vekedag, månad, helg, sesong
- Temperatur-laggar: 1t, 24t, 48t
- Glidande snitt av forbruk: 24t, 48t, 168t
- Laggar av forbruk: 24t og 168t

Dette gir eit godt grunnlag for både forklarande analyse og prognosering.

### Utforskande funn (EDA)

Frå korrelasjon, spreiingsplott og krysskorrelasjon kjem desse hovudfunna fram:

- **Temperatur har sterkast negativ samanheng** med forbruk (kaldare timar gir høgare forbruk).
- **Vindfart har positiv samanheng** med forbruk, truleg via auka oppvarmingsbehov.
- **Soltimar har negativ samanheng** med forbruk.
- **Nedbør har svak/moderat positiv samanheng** med forbruk.
- Krysskorrelasjon indikerer at temperatureffektar varer i om lag **24-48 timar**.

Tidsmønster i forbruket:

- Tydelege toppar om morgonen (ca. 07-09) og ettermiddags/kveldstid (ca. 17-21)
- Høgare nivå i vintermånadene enn i sommarmånadene
- Flatare døgnprofil i helg enn i vekedagar

### Tidsserieanalyse

- ADF-testen for rå timeserie indikerer stasjonaritet (p-verdi < 0,05)
- 1.-ordens differensiering og sesongdifferensiering (lag 24) er likevel brukte for å stabilisere dynamikk og inspeksjon av ACF/PACF
- Additiv dekomponering (periode 24) på eit januarutval viser klart døgnsesongmønster, trendkomponent og residual

### Prognosemodellar

Det er evaluert fire modellar med siste 48 timar som testsett:

- AR
- MA (som ARIMA(0,0,q))
- ARMA
- ARIMA

Orden er vald ved AIC-søk i treningsdata. Notatboka viser samanlikning med MAE, RMSE og MAPE, samt plott av faktisk vs. predikert serie for testperioden. Den faglege oppsummeringa i notatboka peikar på at **ARMA typisk gir best kompromiss** når serien er nær stasjonær, medan ARIMA er nyttig når differensiering trengst.

## 3. Diskusjon

Resultata er konsistente med kjent energiåtferd i bustader: temperatur er den viktigaste drivaren, og døgnrytmen følger hushaldsaktivitet (morgon/kveld). At temperatureffektar varer over fleire timar/døgn støttar valet av lagga feature og tidsseriemodellar med minne.

Samtidig har analysen fleire metodiske avgrensingar:

- **Eitt hushald:** avgrensa generaliserbarheit til heile populasjonen
- **Daglege vêrfelt broadcasta til timar:** reduserer realismen i intradag-effektar for sol og nedbør
- **Univariat modellering:** prognosemodellane nyttar i praksis berre historisk forbruk; eksogene variablar (t.d. temperatur) er ikkje direkte inkluderte i sjølve modellsteget
- **Ingen eksplisitt sesongmodell (SARIMA):** døgnsesong er observert, men ikkje modellert med sesongledd i prognosen

Desse punkta betyr at funna er solide som teknisk demonstrasjon, men at prediksjonskrafta sannsynlegvis kan aukast med sesong- og eksogene modellutvidingar.

## 4. Konklusjon

Analysen dokumenterer ein klar samanheng mellom vêr og straumforbruk for hushaldet `MAC000002`, med temperatur som dominerande faktor. Tidsserien viser stabil døgnstruktur og sesongvariasjon, og klassiske AR/MA-familiar gir eit brukbart grunnlag for 48-timars korttidsprognose.

For vidare arbeid bør ein prioritere:

1. Skalering til fleire hushald for betre robustheit
2. Modellar med eksogene variablar (ARIMAX/SARIMAX)
3. Eksplisitt døgnsesong i prognoseleddet
4. Samanlikning mot maskinlæringsmetodar for lengre horisont

Samla sett gir løysinga ei god, teknisk etterprøvbar plattform for vidare utvikling av datadreven energiprognosering.
