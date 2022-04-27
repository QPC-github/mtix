from .data import *
from mtix.utils import average_top_results, create_lookup, base64_decode, base64_encode
import pytest
from io import StringIO
from unittest import TestCase


XML='<MedlineCitation Status="In-Process" Owner="NLM">\n      <PMID Version="1">35224090</PMID>\n      <DateRevised>\n        <Year>2022</Year>\n        <Month>02</Month>\n        <Day>28</Day>\n      </DateRevised>\n      <Article PubModel="Electronic-eCollection">\n        <Journal>\n          <ISSN IssnType="Electronic">2314-6141</ISSN>\n          <JournalIssue CitedMedium="Internet">\n            <Volume>2022</Volume>\n            <PubDate>\n              <Year>2022</Year>\n            </PubDate>\n          </JournalIssue>\n          <Title>BioMed research international</Title>\n          <ISOAbbreviation>Biomed Res Int</ISOAbbreviation>\n        </Journal>\n        <ArticleTitle>Prolonged Application of Continuous Passive Movement Improves the Postoperative Recovery of Tibial Head Fractures: A Prospective Randomized Controlled Study.</ArticleTitle>\n        <Pagination>\n          <MedlinePgn>1236781</MedlinePgn>\n        </Pagination>\n        <ELocationID EIdType="doi" ValidYN="Y">10.1155/2022/1236781</ELocationID>\n        <Abstract>\n          <AbstractText Label="Methods" NlmCategory="UNASSIGNED">60 patients with THFs were randomly and equally divided into the CPM group and non-CPM group. Both groups immediately received CPM and conventional physical therapies during hospitalization. After discharge, the non-CPM group was treated with conventional physical therapy alone, while the CPM group received conventional physical training in combination with CPM treatment. At 6 weeks and 6 months postoperatively, the primary outcome which was knee ROM and the secondary outcome which was knee functionality and quality of life were evaluated.</AbstractText>\n          <AbstractText Label="Results" NlmCategory="UNASSIGNED">The CPM group had a significantly increased ROM at both follow-up time points. The Knee Society Score, UCLA activity score, and the EuroQoL as well as the pain analysis showed significantly better results of the CPM group than the non-CPM group.</AbstractText>\n          <AbstractText Label="Conclusions" NlmCategory="UNASSIGNED">The prolonged application of CPM therapy is an effective method to improve the postoperative rehabilitation of THFs.</AbstractText>\n          <CopyrightInformation>Copyright © 2022 Christiane Kabst et al.</CopyrightInformation>\n        </Abstract>\n        <AuthorList CompleteYN="Y">\n          <Author ValidYN="Y">\n            <LastName>Kabst</LastName>\n            <ForeName>Christiane</ForeName>\n            <Initials>C</Initials>\n            <AffiliationInfo>\n              <Affiliation>University Center of Orthopaedic, Trauma and Plastic Surgery, University Hospital Carl Gustav Carus at Technische Universität Dresden, 01307 Dresden, Germany.</Affiliation>\n            </AffiliationInfo>\n          </Author>\n          <Author ValidYN="Y">\n            <LastName>Tian</LastName>\n            <ForeName>Xinggui</ForeName>\n            <Initials>X</Initials>\n            <Identifier Source="ORCID">https://orcid.org/0000-0003-3619-3943</Identifier>\n            <AffiliationInfo>\n              <Affiliation>University Center of Orthopaedic, Trauma and Plastic Surgery, University Hospital Carl Gustav Carus at Technische Universität Dresden, 01307 Dresden, Germany.</Affiliation>\n            </AffiliationInfo>\n          </Author>\n          <Author ValidYN="Y">\n            <LastName>Kleber</LastName>\n            <ForeName>Christian</ForeName>\n            <Initials>C</Initials>\n            <AffiliationInfo>\n              <Affiliation>Department of Orthopedics, Trauma Surgery and Plastic Surgery, University Hospital of Leipzig, 04103 Leipzig, Germany.</Affiliation>\n            </AffiliationInfo>\n          </Author>\n          <Author ValidYN="Y">\n            <LastName>Amlang</LastName>\n            <ForeName>Michael</ForeName>\n            <Initials>M</Initials>\n            <AffiliationInfo>\n              <Affiliation>University Center of Orthopaedic, Trauma and Plastic Surgery, University Hospital Carl Gustav Carus at Technische Universität Dresden, 01307 Dresden, Germany.</Affiliation>\n            </AffiliationInfo>\n          </Author>\n          <Author ValidYN="Y">\n            <LastName>Findeisen</LastName>\n            <ForeName>Lisa</ForeName>\n            <Initials>L</Initials>\n            <AffiliationInfo>\n              <Affiliation>University Center of Orthopaedic, Trauma and Plastic Surgery, University Hospital Carl Gustav Carus at Technische Universität Dresden, 01307 Dresden, Germany.</Affiliation>\n            </AffiliationInfo>\n          </Author>\n          <Author ValidYN="Y">\n            <LastName>Lee</LastName>\n            <ForeName>Geoffrey</ForeName>\n            <Initials>G</Initials>\n            <AffiliationInfo>\n              <Affiliation>Kennedy Institute of Rheumatology, Nuffield Department of Orthopedics, Rheumatology, and Musculoskeletal Sciences, University of Oxford, Oxford, UK.</Affiliation>\n            </AffiliationInfo>\n          </Author>\n          <Author ValidYN="Y">\n            <LastName>Zwingenberger</LastName>\n            <ForeName>Stefan</ForeName>\n            <Initials>S</Initials>\n            <AffiliationInfo>\n              <Affiliation>University Center of Orthopaedic, Trauma and Plastic Surgery, University Hospital Carl Gustav Carus at Technische Universität Dresden, 01307 Dresden, Germany.</Affiliation>\n            </AffiliationInfo>\n          </Author>\n        </AuthorList>\n        <Language>eng</Language>\n        <PublicationTypeList>\n          <PublicationType UI="D016428">Journal Article</PublicationType>\n        </PublicationTypeList>\n        <ArticleDate DateType="Electronic">\n          <Year>2022</Year>\n          <Month>02</Month>\n          <Day>16</Day>\n        </ArticleDate>\n      </Article>\n      <MedlineJournalInfo>\n        <Country>United States</Country>\n        <MedlineTA>Biomed Res Int</MedlineTA>\n        <NlmUniqueID>101600173</NlmUniqueID>\n      </MedlineJournalInfo>\n      <CitationSubset>IM</CitationSubset>\n      <CoiStatement>The authors have no conflict of interest to declare.</CoiStatement>\n    </MedlineCitation>\n    '
ENCODED_XML= "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"

LOOKUP_TSV = """1\tD000001
2\tD000002
3\tD000003
4\tD000004
5\tD000005
6\tD000006
7\tD000007
8\tD000008
9\tD000009
10\tD000010
"""
EXPECTED_LOOKUP = {
    1:	"D000001",
    2:	"D000002",
    3:	"D000003",
    4:	"D000004",
    5:	"D000005",
    6:	"D000006",
    7:	"D000007",
    8:	"D000008",
    9:	"D000009",
    10:	"D000010",
}


@pytest.mark.unit
class TestUtils(TestCase):

    def test_base64_encode(self):
        encoded = base64_encode(XML, 9)
        self.assertEqual(encoded, ENCODED_XML, "Incorrect encoding.")

    def test_base64_decode(self):
        decoded = base64_decode(ENCODED_XML)
        self.assertEqual(decoded, XML, "Incorrect decoding.")

    def test_create_lookup(self):
        buffer = StringIO(LOOKUP_TSV)
        lookup = create_lookup(buffer)
        self.assertEqual(lookup, EXPECTED_LOOKUP, "Created lookup not as expected.")

    def test_average_top_results(self):
        results = average_top_results(POINTWISE_AVG_RESULTS, LISTWISE_RESULTS)
        self.assertEqual(results, LISTWISE_AVG_RESULTS, "Average results not as expected.")