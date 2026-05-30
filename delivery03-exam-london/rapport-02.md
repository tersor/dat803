# Analyse av straumforbruk og vêrdata i London

## 1. Introduksjon

Denne rapporten forklarer ein analyse av straumbruk i eitt hus i London.
Vi såg på:

- Kor mykje straum huset brukte kvar time
- Korleis vêret var (temperatur, vind, regn, sol)
- Om vi kan gjette straumbruken dei neste 48 timane

Målet var å forstå kva som påverkar straumbruken mest.

## 2. Resultat

Dei viktigaste funna var:

- Når det er **kaldt**, brukar huset oftast **meir straum**.
- Når det bles meir, går straumbruken ofte litt opp.
- På dagar med meir sol, går straumbruken ofte litt ned.
- Regn ser ut til å gi litt høgare straumbruk.

Vi såg også eit tydeleg mønster i døgnet:

- Høgare bruk om morgonen
- Høgare bruk på ettermiddag/kveld
- Lågare bruk midt på natta

I tillegg prøvde vi fire modellar for å spå straumbruken 48 timar fram i tid.
Alle gav brukbare resultat, og ein kombinasjonsmodell (ARMA) såg best ut i denne analysen.

## 3. Diskusjon

Funna gir god meining i kvardagen:

- Kaldt vêr betyr meir oppvarming
- Folk brukar meir straum når dei er aktive heime (morgon og kveld)

Men det finst nokre avgrensingar:

- Vi brukte berre **eitt hus**. Andre hus kan ha andre vanar.
- Nokre vêrdata var per dag, ikkje per time, så detaljar kan gå tapt.
- Modellane brukte mest tidlegare straumbruk, ikkje alle mogelege forklaringar.

## 4. Konklusjon

Kort oppsummert:

- Vêr, spesielt temperatur, har mykje å seie for straumbruk.
- Straumbruken følgjer eit fast døgnmønster.
- Enkle prognosemodellar kan gi nyttige korttidsestimat.

Dette kan vere nyttig for betre planlegging av energibruk i bustader.
