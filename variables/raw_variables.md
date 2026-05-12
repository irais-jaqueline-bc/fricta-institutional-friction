FRICTA — OFFICIAL COMPUTATIONAL VARIABLES

(Based on survey instrument) ￼

⸻

SECTION 1 — GENERAL INFORMATION

⸻

Q1 — Estado

QUESTION

Estado ￼

⸻

VARIABLE

state

⸻

TYPE

categorical_nominal

⸻

ENCODING

String categorical:

CDMX
Jalisco
Puebla etc.
⸻

BRANCH

metadata

⸻

PURPOSE

Institutional geographic diversity.

⸻

Q2 — Tipo de institución

QUESTION

¿Qué tipo de institución es? ￼

⸻

VARIABLE

institution_type

⸻

TYPE

categorical_nominal

⸻

ENCODING

respuesta valor Pública public Privada private Asociación civil / ONG ngo Mixta mixed No estoy seguro unknown

⸻

BRANCH

Organizational Constraints

⸻

Q3 — Número de niños atendidos

VARIABLE

children_served

⸻

TYPE

ordinal

⸻

ENCODING

respuesta valor 1-10 1 11-30 2 31-60 3 60+ 4

⸻

BRANCH

metadata

⸻

Q4 — Número aproximado de empleados

VARIABLE

staff_size

⸻

TYPE

ordinal

⸻

ENCODING

respuesta valor 1-5 1 6-15 2 16-30 3 30+ 4

⸻

BRANCH

Operational Load

⸻

SECTION 2 — DIGITAL INFRASTRUCTURE

⸻

Q5 — Computadoras disponibles

VARIABLE

available_devices

⸻

TYPE

ordinal

⸻

ENCODING

respuesta valor 0 0 1-2 1 3-5 2 6+ 3

⸻

BRANCH

Infrastructure Constraints

⸻

Q6 — Estabilidad de internet

VARIABLE

internet_stability

⸻

TYPE

ordinal

⸻

ENCODING

respuesta valor Muy estable 5 Estable 4 Inestable 3 Muy inestable 2 No hay acceso 1

⸻

BRANCH

Infrastructure Constraints

⸻

Q7 — Herramientas digitales utilizadas

VARIABLE

digital_tools_used

⸻

TYPE

multi_label_categorical

⸻

ENCODING

Binary encoding:

herramienta variable Excel uses_excel WhatsApp uses_whatsapp Google Drive / Docs uses_google_workspace Software especializado uses_specialized_software Ninguna no_digital_tools Otra other_tool

⸻

DERIVED VARIABLE

digital_tool_variety

⸻

FORMULA

DigitalToolVariety = \sum Tool_i

⸻

BRANCH

Existing Digital Integration

⸻

SECTION 3 — ADMINISTRATIVE PROCESSES

⸻

Q8 — Registro de información

VARIABLE

registration_system_type

⸻

TYPE

ordinal

⸻

ENCODING

respuesta valor Papel 1 Excel 2 Mixto 3 Software 4

⸻

BRANCH

Existing Digital Integration

⸻

Q9 — Tiempo administrativo diario

VARIABLE

admin_time_load

⸻

TYPE

ordinal

⸻

ENCODING

respuesta valor Menos de 1 hora 1 1-3 horas 2 3-5 horas 3 5+ horas 4

⸻

BRANCH

Operational Load

⸻

Q10 — Organización administrativa percibida

VARIABLE

administrative_organization

⸻

TYPE

ordinal

⸻

ENCODING

respuesta valor Muy alto 5 Alto 4 Medio 3 Bajo 2 Muy bajo 1

⸻

BRANCH

Organizational Constraints

⸻

SECTION 4 — DIGITAL ADOPTION

⸻

Q11 — Frecuencia de uso digital

VARIABLE

digital_usage_frequency

⸻

TYPE

ordinal

⸻

ENCODING

respuesta valor Diario 4 Semanal 3 Rara vez 2 Nunca 1

⸻

BRANCH

Existing Digital Integration

⸻

Q12 — Intentos previos de implementación

VARIABLE

previous_digital_implementation

⸻

TYPE

binary

⸻

ENCODING

respuesta valor Sí 1 No 0

⸻

BRANCH

Organizational Constraints

⸻

Q13 — Dificultad de implementación

VARIABLE

implementation_difficulty

⸻

TYPE

ordinal

⸻

ENCODING

respuesta valor Muy fácil 1 Fácil 2 Moderado 3 Difícil 4 Muy difícil 5

⸻

BRANCH

Organizational Constraints

⸻

SECTION 5 — FRICTION

⸻

Q14A — Falta de tiempo

VARIABLE

time_constraint

⸻

TYPE

ordinal

⸻

ENCODING

respuesta valor Nada 1 Poco 2 Medio 3 Alto 4 Muy alto 5

⸻

BRANCH

Operational Load

⸻

Q14B — Falta de personal

VARIABLE

staffing_constraint

⸻

BRANCH

Operational Load

⸻

Q14C — Falta de capacitación

VARIABLE

training_deficit

⸻

BRANCH

Human Capacity & Training

⸻

Q14D — Falta de recursos

VARIABLE

resource_constraint

⸻

BRANCH

Infrastructure Constraints

⸻

ENCODING FOR ALL Q14

respuesta valor Nada 1 Poco 2 Medio 3 Alto 4 Muy alto 5

⸻

Q15 — Dificultad de cambiar sistema

VARIABLE

system_change_resistance

⸻

TYPE

ordinal

⸻

ENCODING

respuesta valor Muy fácil 1 Fácil 2 Neutral 3 Difícil 4 Muy difícil 5

⸻

BRANCH

Organizational Constraints

⸻

Q16 — Utilidad percibida

VARIABLE

perceived_digital_utility

⸻

TYPE

ordinal

⸻

ENCODING

respuesta valor Mucho 4 Algo 3 Poco 2 Nada 1

⸻

BRANCH

Existing Digital Integration

⸻

Q17 — Disposición a probar herramienta

VARIABLE

tool_adoption_willingness

⸻

TYPE

ordinal

⸻

ENCODING

respuesta valor Muy dispuestos 5 Dispuestos 4 Neutrales 3 Poco dispuestos 2 Nada dispuestos 1

⸻

BRANCH

Human Capacity & Training

⸻

Q18 — Apertura a piloto

VARIABLE

pilot_openness

⸻

TYPE

ordinal

⸻

ENCODING

respuesta valor Sí 3 Tal vez 2 No 1

⸻

BRANCH

metadata