"""
원소/동위원소 데이터베이스 및 기본 타입 정의.

SCIEX Clearcore2.ElementsAndMolecules 에서 이식:
- Element, MonoIsotopicElement, MultiIsotopicElement
- ElementCount, FormulaUtils 일부
"""

from __future__ import annotations
import re
import math
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
#  상수
# ═══════════════════════════════════════════════════════════════
ELECTRON_MASS = 0.000548579909  # Da

# 할로겐 원소 원자번호
HALOGEN_ATOMIC_NUMBERS = frozenset({9, 17, 35, 53, 85})  # F, Cl, Br, I, At

# 생체 원소 원자번호
BIO_ELEMENT_ATOMIC_NUMBERS = frozenset({6, 1, 7, 8, 15, 16})  # C, H, N, O, P, S


# ═══════════════════════════════════════════════════════════════
#  기본 타입
# ═══════════════════════════════════════════════════════════════

@dataclass
class Isotope:
    """단일 동위원소 (MonoIsotopicElement 대응)"""
    mass: float
    abundance: float  # 0~1 정규화된 존재비
    nominal_mass: int = 0

    def __post_init__(self):
        self.nominal_mass = int(self.mass + 0.5)


@dataclass
class Element:
    """
    원소 정의 (MultiIsotopicElement 대응).
    isotopes: [(mass, raw_abundance), ...] 순서대로.
    """
    symbol: str
    name: str
    atomic_number: int
    default_valence: int
    isotopes: list[Isotope] = field(default_factory=list)
    display_order: float = 0.0

    # 캐시
    _monoisotopic_mass: Optional[float] = field(default=None, repr=False)
    _average_mass: Optional[float] = field(default=None, repr=False)

    @property
    def monoisotopic_mass(self) -> float:
        """가장 존재비 높은 동위원소의 질량"""
        if self._monoisotopic_mass is None:
            best = max(self.isotopes, key=lambda iso: iso.abundance)
            self._monoisotopic_mass = best.mass
        return self._monoisotopic_mass

    @property
    def average_mass(self) -> float:
        """가중평균 질량"""
        if self._average_mass is None:
            total_ab = sum(iso.abundance for iso in self.isotopes)
            if total_ab == 0:
                self._average_mass = 0.0
            else:
                self._average_mass = sum(iso.mass * iso.abundance for iso in self.isotopes) / total_ab
        return self._average_mass

    @property
    def nominal_mass(self) -> int:
        return int(self.monoisotopic_mass + 0.5)

    @property
    def is_halogen(self) -> bool:
        return self.atomic_number in HALOGEN_ATOMIC_NUMBERS

    @property
    def is_common_bio_element(self) -> bool:
        return self.atomic_number in BIO_ELEMENT_ATOMIC_NUMBERS

    @property
    def num_isotopes(self) -> int:
        return len(self.isotopes)


@dataclass
class ElementCount:
    """원소-개수 쌍 (ElementCount 대응)"""
    element: Element
    count: int

    def __hash__(self):
        return hash((self.element.symbol, self.count))

    def __eq__(self, other):
        if not isinstance(other, ElementCount):
            return False
        return self.element.symbol == other.element.symbol and self.count == other.count


# ═══════════════════════════════════════════════════════════════
#  주기율표 데이터
#  SCIEX ChemicalElements.xml (NIST 2016-05-26) 기준
#  format: (symbol, name, atomic_number, default_valence, display_order,
#           [(mass, abundance), ...])
# ═══════════════════════════════════════════════════════════════

_ELEMENTS_DATA = [
    # ── SCIEX ChemicalElements.xml (NIST 2016-05-26) 기준 ──
    ("H", "Hydrogen", 1, 1, 1.0, [
        (1.00782503223, 0.999885),
        (2.01410177812, 0.000115),
    ]),
    ("D", "Deuterium", 1, 1, 1.5, [
        (2.01410177812, 1.0),
    ]),
    ("He", "Helium", 2, 0, 2.0, [
        (3.0160293201, 0.00000134),
        (4.0026032541, 0.99999866),
    ]),
    ("Li", "Lithium", 3, 1, 3.0, [
        (6.0151228874, 0.0759),
        (7.0160034366, 0.9241),
    ]),
    ("Be", "Beryllium", 4, 2, 4.0, [
        (9.0121830650, 1.0),
    ]),
    ("B", "Boron", 5, 3, 5.0, [
        (10.0129369500, 0.199),
        (11.0093053600, 0.801),
    ]),
    ("C", "Carbon", 6, 4, 6.0, [
        (12.0, 0.9893),
        (13.00335483507, 0.0107),
    ]),
    ("N", "Nitrogen", 7, 3, 7.0, [
        (14.00307400443, 0.99636),
        (15.00010889888, 0.00364),
    ]),
    ("O", "Oxygen", 8, 2, 8.0, [
        (15.99491461957, 0.99757),
        (16.99913175650, 0.00038),
        (17.99915961286, 0.00205),
    ]),
    ("F", "Fluorine", 9, 1, 9.0, [
        (18.99840316273, 1.0),
    ]),
    ("Ne", "Neon", 10, 0, 10.0, [
        (19.99244017620, 0.9048),
        (20.99384668500, 0.0027),
        (21.991385114, 0.0925),
    ]),
    ("Na", "Sodium", 11, 1, 11.0, [
        (22.98976928200, 1.0),
    ]),
    ("Mg", "Magnesium", 12, 2, 12.0, [
        (23.98504169700, 0.7899),
        (24.98583697600, 0.1000),
        (25.98259296800, 0.1101),
    ]),
    ("Al", "Aluminium", 13, 3, 13.0, [
        (26.98153853, 1.0),
    ]),
    ("Si", "Silicon", 14, 4, 14.0, [
        (27.97692653465, 0.92223),
        (28.97649466490, 0.04685),
        (29.97377013600, 0.03092),
    ]),
    ("P", "Phosphorus", 15, 3, 15.0, [
        (30.97376199842, 1.0),
    ]),
    ("S", "Sulfur", 16, 2, 16.0, [
        (31.97207117440, 0.9499),
        (32.97145890980, 0.0075),
        (33.96786700400, 0.0425),
        (35.96708071, 0.0001),
    ]),
    ("Cl", "Chlorine", 17, 1, 17.0, [
        (34.96885268200, 0.7576),
        (36.96590260200, 0.2424),
    ]),
    ("Ar", "Argon", 18, 0, 18.0, [
        (35.96754510500, 0.003336),
        (37.96273211, 0.000629),
        (39.96238312370, 0.996035),
    ]),
    ("K", "Potassium", 19, 1, 19.0, [
        (38.96370648640, 0.932581),
        (39.96399816600, 0.000117),
        (40.96182525790, 0.067302),
    ]),
    ("Ca", "Calcium", 20, 2, 20.0, [
        (39.96259086300, 0.96941),
        (41.95861783, 0.00647),
        (42.95876644, 0.00135),
        (43.95548156, 0.02086),
        (45.953689, 0.00004),
        (47.95252276, 0.00187),
    ]),
    ("Ti", "Titanium", 22, 2, 22.0, [
        (45.95262772, 0.0825),
        (46.95175879, 0.0744),
        (47.94794198, 0.7372),
        (48.94786568, 0.0541),
        (49.94478689, 0.0518),
    ]),
    ("V", "Vanadium", 23, 2, 23.0, [
        (49.94715601, 0.00250),
        (50.94395704, 0.99750),
    ]),
    ("Cr", "Chromium", 24, 2, 24.0, [
        (49.94604183, 0.04345),
        (51.94050623, 0.83789),
        (52.94064815, 0.09501),
        (53.93887916, 0.02365),
    ]),
    ("Mn", "Manganese", 25, 2, 25.0, [
        (54.93804391, 1.0),
    ]),
    ("Fe", "Iron", 26, 2, 26.0, [
        (53.93960899, 0.05845),
        (55.93493633, 0.91754),
        (56.93539284, 0.02119),
        (57.93327443, 0.00282),
    ]),
    ("Co", "Cobalt", 27, 2, 27.0, [
        (58.93319429, 1.0),
    ]),
    ("Ni", "Nickel", 28, 3, 28.0, [
        (57.93534241, 0.68077),
        (59.93078588, 0.26223),
        (60.93105557, 0.011399),
        (61.92834537, 0.036346),
        (63.92796682, 0.009255),
    ]),
    ("Cu", "Copper", 29, 2, 29.0, [
        (62.92959772, 0.6915),
        (64.92778970, 0.3085),
    ]),
    ("Zn", "Zinc", 30, 2, 30.0, [
        (63.92914201, 0.4917),
        (65.92603381, 0.2773),
        (66.92712775, 0.0404),
        (67.92484455, 0.1845),
        (69.9253192, 0.0061),
    ]),
    ("Ga", "Gallium", 31, 3, 31.0, [
        (68.9255735, 0.60108),
        (70.92470258, 0.39892),
    ]),
    ("Ge", "Germanium", 32, 2, 32.0, [
        (69.92424875, 0.2057),
        (71.922075826, 0.2745),
        (72.923458956, 0.0775),
        (73.921177761, 0.3650),
        (75.921402726, 0.0773),
    ]),
    ("As", "Arsenic", 33, 0, 33.0, [
        (74.92159457, 1.0),
    ]),
    ("Se", "Selenium", 34, 4, 34.0, [
        (73.922475934, 0.0089),
        (75.919213704, 0.0937),
        (76.919914154, 0.0763),
        (77.91730928, 0.2377),
        (79.9165218, 0.4961),
        (81.9166995, 0.0873),
    ]),
    ("Br", "Bromine", 35, 1, 35.0, [
        (78.9183376, 0.5069),
        (80.9162897, 0.4931),
    ]),
    ("Kr", "Krypton", 36, 0, 36.0, [
        (77.92036494, 0.00355),
        (79.91637808, 0.02286),
        (81.91348273, 0.11593),
        (82.91412716, 0.11500),
        (83.9114977282, 0.56987),
        (85.9106106269, 0.17279),
    ]),
    ("Rb", "Rubidium", 37, 2, 37.0, [
        (84.9117897379, 0.7217),
        (86.9091805310, 0.2783),
    ]),
    ("Sr", "Strontium", 38, 2, 38.0, [
        (83.9134191, 0.0056),
        (85.9092606, 0.0986),
        (86.9088775, 0.0700),
        (87.9056125, 0.8258),
    ]),
    ("Zr", "Zirconium", 40, 4, 40.0, [
        (89.9046977, 0.5145),
        (90.9056396, 0.1122),
        (91.9050347, 0.1715),
        (93.9063108, 0.1738),
        (95.9082714, 0.0280),
    ]),
    ("Mo", "Molybdenum", 42, 2, 42.0, [
        (91.90680796, 0.1453),
        (93.90508490, 0.0915),
        (94.90583877, 0.1584),
        (95.90467612, 0.1667),
        (96.90601812, 0.0960),
        (97.90540482, 0.2439),
        (99.9074718, 0.0982),
    ]),
    ("Ru", "Ruthenium", 44, 0, 44.0, [
        (95.90759025, 0.0554),
        (97.9052868, 0.0187),
        (98.9059341, 0.1276),
        (99.9042143, 0.1260),
        (100.9055769, 0.1706),
        (101.9043441, 0.3155),
        (103.9054275, 0.1862),
    ]),
    ("Rh", "Rhodium", 45, 2, 45.0, [
        (102.9054980, 1.0),
    ]),
    ("Pd", "Palladium", 46, 2, 46.0, [
        (101.9056022, 0.0102),
        (103.9040305, 0.1114),
        (104.9050796, 0.2233),
        (105.9034804, 0.2733),
        (107.9038916, 0.2646),
        (109.90517220, 0.1172),
    ]),
    ("Ag", "Silver", 47, 1, 47.0, [
        (106.9050916, 0.51839),
        (108.9047553, 0.48161),
    ]),
    ("Cd", "Cadmium", 48, 2, 48.0, [
        (105.9064599, 0.0125),
        (107.9041834, 0.0089),
        (109.90300661, 0.1249),
        (110.90418287, 0.1280),
        (111.90276287, 0.2413),
        (112.90440813, 0.1222),
        (113.90336509, 0.2873),
        (115.90476315, 0.0749),
    ]),
    ("In", "Indium", 49, 3, 49.0, [
        (112.90406184, 0.0429),
        (114.903878776, 0.9571),
    ]),
    ("Sn", "Tin", 50, 4, 50.0, [
        (111.90482387, 0.0097),
        (113.9027827, 0.0066),
        (114.903344699, 0.0034),
        (115.90174280, 0.1454),
        (116.90295398, 0.0768),
        (117.90160657, 0.2422),
        (118.90331117, 0.0859),
        (119.90220163, 0.3258),
        (121.9034438, 0.0463),
        (123.9052766, 0.0579),
    ]),
    ("Sb", "Antimony", 51, 0, 51.0, [
        (120.9038120, 0.5721),
        (122.9042132, 0.4279),
    ]),
    ("Te", "Tellurium", 52, 2, 52.0, [
        (119.9040593, 0.0009),
        (121.9030435, 0.0255),
        (122.9042698, 0.0089),
        (123.9028171, 0.0474),
        (124.9044299, 0.0707),
        (125.9033109, 0.1884),
        (127.90446128, 0.3174),
        (129.906222748, 0.3408),
    ]),
    ("I", "Iodine", 53, 1, 53.0, [
        (126.9044719, 1.0),
    ]),
    ("Xe", "Xenon", 54, 0, 54.0, [
        (123.905892, 0.000952),
        (125.9042983, 0.000890),
        (127.9035310, 0.019102),
        (128.90478086110, 0.264006),
        (129.903509349, 0.040710),
        (130.90508406, 0.212324),
        (131.9041550856, 0.269086),
        (133.90539466, 0.104357),
        (135.907214484, 0.088573),
    ]),
    ("Cs", "Caesium", 55, 1, 55.0, [
        (132.9054519610, 1.0),
    ]),
    ("Ba", "Barium", 56, 2, 56.0, [
        (129.9063207, 0.00106),
        (131.9050611, 0.00101),
        (133.90450818, 0.02417),
        (134.90568838, 0.06592),
        (135.90457573, 0.07854),
        (136.90582714, 0.11232),
        (137.90524700, 0.71698),
    ]),
    ("La", "Lanthanum", 57, 3, 57.0, [
        (137.9071149, 0.0008881),
        (138.9063563, 0.9991119),
    ]),
    ("W", "Tungsten", 74, 5, 74.0, [
        (179.9467108, 0.0012),
        (181.94820394, 0.2650),
        (182.95022275, 0.1431),
        (183.95093092, 0.3064),
        (185.95436280, 0.2843),
    ]),
    ("Re", "Rhenium", 75, 2, 75.0, [
        (184.9529545, 0.3740),
        (186.9557501, 0.6260),
    ]),
    ("Os", "Osmium", 76, 4, 76.0, [
        (183.9524885, 0.0002),
        (185.9538350, 0.0159),
        (186.9557474, 0.0196),
        (187.9558352, 0.1324),
        (188.9581442, 0.1615),
        (189.9584437, 0.2626),
        (191.9614770, 0.4078),
    ]),
    ("Ir", "Iridium", 77, 4, 77.0, [
        (190.9605893, 0.373),
        (192.9629216, 0.627),
    ]),
    ("Pt", "Platinum", 78, 4, 78.0, [
        (189.9599297, 0.00012),
        (191.9610387, 0.00782),
        (193.9626809, 0.3286),
        (194.9647917, 0.3378),
        (195.96495209, 0.2521),
        (197.9678949, 0.07356),
    ]),
    ("Au", "Gold", 79, 3, 79.0, [
        (196.96656879, 1.0),
    ]),
    ("Hg", "Mercury", 80, 2, 80.0, [
        (195.9658326, 0.0015),
        (197.96676860, 0.0997),
        (198.96828064, 0.1687),
        (199.96832659, 0.2310),
        (200.97030284, 0.1318),
        (201.97064340, 0.2986),
        (203.97349398, 0.0687),
    ]),
    ("Tl", "Thallium", 81, 3, 81.0, [
        (202.9723446, 0.2952),
        (204.9744278, 0.7048),
    ]),
    ("Pb", "Lead", 82, 2, 82.0, [
        (203.9730440, 0.014),
        (205.9744657, 0.241),
        (206.9758973, 0.221),
        (207.9766525, 0.524),
    ]),
    ("Bi", "Bismuth", 83, 3, 83.0, [
        (208.9803991, 1.0),
    ]),
    ("U", "Uranium", 92, 6, 92.0, [
        (234.0409523, 0.000054),
        (235.0439301, 0.007204),
        (238.0507884, 0.992742),
    ]),
]


# ═══════════════════════════════════════════════════════════════
#  원소 데이터베이스
# ═══════════════════════════════════════════════════════════════

class PeriodicTable:
    """원소 데이터베이스 싱글톤"""

    _instance: Optional[PeriodicTable] = None

    def __init__(self):
        self._by_symbol: dict[str, Element] = {}
        self._by_atomic_number: dict[int, Element] = {}
        self._load_defaults()

    @classmethod
    def instance(cls) -> PeriodicTable:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_defaults(self):
        for sym, name, anum, valence, dorder, iso_data in _ELEMENTS_DATA:
            total_ab = sum(ab for _, ab in iso_data)
            isotopes = []
            for mass, ab in iso_data:
                isotopes.append(Isotope(mass=mass, abundance=ab / total_ab if total_ab > 0 else 0.0))
            elem = Element(
                symbol=sym,
                name=name,
                atomic_number=anum,
                default_valence=valence,
                isotopes=isotopes,
                display_order=dorder,
            )
            self._by_symbol[sym] = elem
            # D (Deuterium)은 atomic_number=1이지만 H가 이미 있으므로 덮어쓰지 않음
            if anum not in self._by_atomic_number:
                self._by_atomic_number[anum] = elem

    def get(self, symbol: str) -> Optional[Element]:
        return self._by_symbol.get(symbol)

    def get_by_atomic_number(self, anum: int) -> Optional[Element]:
        return self._by_atomic_number.get(anum)

    def add_custom_element(self, elem: Element):
        self._by_symbol[elem.symbol] = elem

    @property
    def all_elements(self) -> list[Element]:
        return list(self._by_symbol.values())

    @property
    def symbols(self) -> list[str]:
        return sorted(self._by_symbol.keys(), key=lambda s: (-len(s), s))


# ═══════════════════════════════════════════════════════════════
#  분자식 파싱
# ═══════════════════════════════════════════════════════════════

# 분자식 파싱 정규식: 원소기호(대문자+소문자*) + 선택적 숫자
_FORMULA_RE = re.compile(r'([A-Z][a-z]?)(\d*)')

def parse_formula(formula: str, pt: Optional[PeriodicTable] = None) -> list[ElementCount]:
    """
    분자식 문자열을 ElementCount 리스트로 변환.
    예: "C6H12O6" -> [ElementCount(C,6), ElementCount(H,12), ElementCount(O,6)]
    """
    if pt is None:
        pt = PeriodicTable.instance()

    counts: dict[str, int] = {}
    for match in _FORMULA_RE.finditer(formula.strip()):
        sym = match.group(1)
        num = int(match.group(2)) if match.group(2) else 1
        if sym in counts:
            counts[sym] += num
        else:
            counts[sym] = num

    result = []
    for sym, cnt in counts.items():
        elem = pt.get(sym)
        if elem is None:
            raise ValueError(f"Unknown element symbol: {sym}")
        result.append(ElementCount(element=elem, count=cnt))

    return result


def formula_to_string(element_counts: list[ElementCount]) -> str:
    """ElementCount 리스트 → 분자식 문자열 (Hill 순서: C, H, 나머지 알파벳)"""
    if not element_counts:
        return ""

    # Hill 순서 정렬
    def hill_key(ec: ElementCount) -> tuple:
        sym = ec.element.symbol
        if sym == "C":
            return (0, sym)
        elif sym == "H":
            return (1, sym)
        else:
            return (2, sym)

    sorted_ecs = sorted(element_counts, key=hill_key)
    parts = []
    for ec in sorted_ecs:
        if ec.count == 0:
            continue
        if ec.count == 1:
            parts.append(ec.element.symbol)
        else:
            parts.append(f"{ec.element.symbol}{ec.count}")
    return "".join(parts)


# ═══════════════════════════════════════════════════════════════
#  기본 계산 유틸리티 (FormulaUtils 대응)
# ═══════════════════════════════════════════════════════════════

def get_monoisotopic_mass(element_counts: list[ElementCount]) -> float:
    """원소 조합의 monoisotopic mass 계산"""
    return sum(ec.element.monoisotopic_mass * ec.count for ec in element_counts)


def get_average_mass(element_counts: list[ElementCount]) -> float:
    """원소 조합의 average mass 계산"""
    return sum(ec.element.average_mass * ec.count for ec in element_counts)


def get_nominal_mass(element_counts: list[ElementCount]) -> int:
    """원소 조합의 nominal mass 계산"""
    return sum(ec.element.nominal_mass * ec.count for ec in element_counts)


def calculate_rdb(element_counts: list[ElementCount]) -> float:
    """
    Ring and Double Bond equivalents (RDB/DBE) 계산.
    RDB = 1 + Σ(count_i × (valence_i/2 - 1))
    """
    rdb = 1.0
    for ec in element_counts:
        rdb += ec.count * (ec.element.default_valence / 2.0 - 1.0)
    return rdb


def contains_halogen(element_counts: list[ElementCount]) -> bool:
    return any(ec.element.is_halogen for ec in element_counts)


def combine_element_counts(element_counts: list[ElementCount]) -> list[ElementCount]:
    """동일 원소를 합산"""
    combined: dict[str, ElementCount] = {}
    for ec in element_counts:
        sym = ec.element.symbol
        if sym in combined:
            combined[sym] = ElementCount(ec.element, combined[sym].count + ec.count)
        else:
            combined[sym] = ElementCount(ec.element, ec.count)
    return [ec for ec in combined.values() if ec.count != 0]
