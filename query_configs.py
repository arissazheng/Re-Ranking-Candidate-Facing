from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class QueryConfig:
    """In-memory representation of a single evaluation query."""

    name: str
    description: str
    hard_criteria: str
    soft_criteria: str
    config_path: str


def get_all_query_configs() -> List[QueryConfig]:
    """Returns the 10 public evaluation query configs.

    These are transcribed from the provided evaluation sheet.
    """
    return [
        QueryConfig(
            name="Tax Lawyer",
            description=(
                "Seasoned attorney with a JD from a top U.S. law school and over three "
                "years of legal practice, specializing in corporate tax structuring and "
                "compliance. Has represented clients in IRS audits and authored legal "
                "opinions on federal tax code matters."
            ),
            hard_criteria=(
                "1. JD degree from an accredited U.S. law school\n"
                "2. 3+ years of experience practicing law"
            ),
            soft_criteria=(
                "1. Experience advising clients on tax implications of corporate or financial transactions\n"
                "2. Experience handling IRS audits, disputes, or regulatory inquiries\n"
                "3. Experience drafting legal opinions or filings related to federal and state tax compliance"
            ),
            config_path="tax_lawyer.yml",
        ),
        QueryConfig(
            name="Junior Corporate Lawyer",
            description=(
                "Corporate lawyer with two years of experience at a top-tier international law firm, "
                "specializing in M&A support and cross-border contract negotiations. Trained at a leading "
                "European law school with additional background in international regulatory compliance."
            ),
            hard_criteria=(
                "1. 2-4 years of experience as a Corporate Lawyer at a leading law firm in the USA, Europe, or Canada, "
                "or in-house at a major global organization\n"
                "2. Graduate of a reputed law school in the USA, Europe, or Canada"
            ),
            soft_criteria=(
                "1. Experience supporting Corporate M&A transactions, including due diligence and legal documentation\n"
                "2. Experience drafting and negotiating legal contracts or commercial agreements\n"
                "3. Familiarity with international business law or advising on regulatory requirements across jurisdictions"
            ),
            config_path="junior_corporate_lawyer.yml",
        ),
        QueryConfig(
            name="Radiology",
            description=(
                "Radiologist with an MD from India and several years of experience reading CT and MRI scans. "
                "Well-versed in diagnostic workflows and has worked on projects involving AI-assisted image analysis."
            ),
            hard_criteria=(
                "1. MD degree from a medical school in the U.S. or India"
            ),
            soft_criteria=(
                "1. Board certification in Radiology (ABR, FRCR, or equivalent) or comparable credential\n"
                "2. 3+ years of experience interpreting X-ray, CT, MRI, ultrasound, or nuclear medicine studies\n"
                "3. Expertise in radiology reporting, diagnostic protocols, differential diagnosis, or AI applications in medical imaging"
            ),
            config_path="radiology.yml",
        ),
        QueryConfig(
            name="Doctors (MD)",
            description=(
                "U.S.-trained physician with over two years of experience as a general practitioner, "
                "focused on chronic care management, wellness screenings, and outpatient diagnostics. "
                "Skilled in telemedicine and patient education."
            ),
            hard_criteria=(
                "1. MD degree from a top U.S. medical school\n"
                "2. 2+ years of clinical practice experience in the U.S.\n"
                "3. Experience working as a General Practitioner (GP)"
            ),
            soft_criteria=(
                "1. Familiarity with EHR systems and managing high patient volumes in outpatient or family medicine settings\n"
                "2. Comfort with telemedicine consultations, patient triage, and interdisciplinary coordination"
            ),
            config_path="doctors_md.yml",
        ),
        QueryConfig(
            name="Biology Expert",
            description=(
                "Biologist with a PhD from a top U.S. university, specializing in molecular biology and gene expression."
            ),
            hard_criteria=(
                "1. Completed undergraduate studies in the U.S., U.K., or Canada\n"
                "2. PhD in Biology from a top U.S. university"
            ),
            soft_criteria=(
                "1. Research experience in molecular biology, genetics, or cell biology, with publications in peer-reviewed journals\n"
                "2. Familiarity with experimental design, data analysis, and lab techniques such as CRISPR, PCR, or sequencing\n"
                "3. Experience mentoring students, teaching undergraduate biology courses, or collaborating on interdisciplinary research"
            ),
            config_path="biology_expert.yml",
        ),
        QueryConfig(
            name="Anthropology",
            description=(
                "PhD student in anthropology at a top U.S. university, focused on labor migration and cultural identity."
            ),
            hard_criteria=(
                "1. PhD (in progress or completed) from a distinguished program in sociology, anthropology, or economics\n"
                "2. PhD program started within the last 3 years"
            ),
            soft_criteria=(
                "1. Demonstrated expertise in ethnographic methods, with substantial fieldwork or case study research involving cultural, social, or economic systems\n"
                "2. Strong academic output — published papers, working papers, or conference presentations on anthropological or sociological topics\n"
                "3. Experience applying anthropological theory to real-world or interdisciplinary contexts (e.g., migration, labor, technology, development)"
            ),
            config_path="anthropology.yml",
        ),
        QueryConfig(
            name="Mathematics PhD",
            description=(
                "Mathematician with a PhD from a leading U.S. university, specializing in statistical inference and "
                "stochastic processes. Published and experienced in both theoretical and applied research."
            ),
            hard_criteria=(
                "1. Completed undergraduate studies in the U.S., U.K., or Canada\n"
                "2. PhD in Mathematics or Statistics from a top U.S. university"
            ),
            soft_criteria=(
                "1. Research expertise in pure or applied mathematics, statistics, or probability, with peer-reviewed publications or preprints\n"
                "2. Proficiency in mathematical modeling, proof-based reasoning, or algorithmic problem-solving"
            ),
            config_path="mathematics_phd.yml",
        ),
        QueryConfig(
            name="Quantitative Finance",
            description=(
                "MBA graduate from a top U.S. program with 3+ years of experience in quantitative finance, including "
                "roles in risk modeling and algorithmic trading at a global investment firm. Skilled in Python and "
                "financial modeling, with expertise in portfolio optimization and derivatives pricing."
            ),
            hard_criteria=(
                "1. MBA from a Prestigious U.S. university (M7 MBA)\n"
                "2. 3+ years of experience in quantitative finance, including roles such as risk modeling, algorithmic trading, or financial engineering"
            ),
            soft_criteria=(
                "1. Experience applying financial modeling techniques to real-world problems like portfolio optimization or derivatives pricing\n"
                "2. Proficiency with Python for quantitative analysis and exposure to financial libraries (e.g., QuantLib or equivalent)\n"
                "3. Demonstrated ability to work in high-stakes environments such as global investment firms, showing applied knowledge of quantitative methods in production settings"
            ),
            config_path="quantitative_finance.yml",
        ),
        QueryConfig(
            name="Bankers",
            description=(
                "Healthcare investment banker with over two years at a leading advisory firm, focused on M&A for "
                "multi-site provider groups and digital health companies. Currently working in a healthcare-focused "
                "growth equity fund, driving diligence and investment strategy."
            ),
            hard_criteria=(
                "1. MBA from a U.S. university\n"
                "2. 2+ years of prior work experience in investment banking, corporate finance, or M&A advisory"
            ),
            soft_criteria=(
                "1. Specialized experience in healthcare-focused investment banking or private equity, including exposure to sub-verticals like biotech, pharma services, or provider networks\n"
                "2. Led or contributed to transactions involving healthcare M&A, recapitalizations, or growth equity investments\n"
                "3. Familiarity with healthcare-specific metrics, regulatory frameworks, and value creation strategies (e.g., payer-provider integration, RCM optimization)"
            ),
            config_path="bankers.yml",
        ),
        QueryConfig(
            name="Mechanical Engineers",
            description=(
                "Mechanical engineer with over three years of experience in product development and structural design, "
                "using tools like SolidWorks and ANSYS. Led thermal system simulations and supported prototyping for "
                "electromechanical components in an industrial R&D setting."
            ),
            hard_criteria=(
                "1. Higher degree in Mechanical Engineering from an accredited university\n"
                "2. 3+ years of professional experience in mechanical design, product development, or systems engineering"
            ),
            soft_criteria=(
                "1. Experience with CAD tools (e.g., SolidWorks, AutoCAD) and mechanical simulation tools (e.g., ANSYS, COMSOL)\n"
                "2. Demonstrated involvement in end-to-end product lifecycle — from concept through prototyping to manufacturing or testing\n"
                "3. Domain specialization in areas like thermal systems, fluid dynamics, structural analysis, or mechatronics"
            ),
            config_path="mechanical_engineers.yml",
        ),
    ]


