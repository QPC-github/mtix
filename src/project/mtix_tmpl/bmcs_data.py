import json
import base64
import gzip
import xml.etree.ElementTree as ET


class BmcsData(object):
    def __init__(self, pmid: int = None, rid: int = None, xml: str = None):
        self._pmid = pmid if pmid is not None else None
        self._rid = rid if rid is not None else None
        self._xml = xml if xml is not None else None

    def pmid(self, pmid: int = None):
        if pmid is not None:
            self._pmid = pmid
        return self._pmid

    def rid(self, rid: int = None):
        if rid is not None:
            self._rid = rid
        return self._rid

    def xml(self, xml: str = None):
        if xml is not None:
            self._xml = xml
        return self._xml

    def to_bytes(self):
        return "|".join([self._rid, self._pmid, self._xml]).encode("utf-8")

    @classmethod
    def from_bytes(cls, lBytes):
        rid, pmid, xml = lBytes.decode("utf-8").split('|', 3)
        return cls(rid=rid, pmid=pmid, xml=xml)

    def __str__(self):
        return "{}|{}|{}".format(self._rid, self._pmid, self._xml)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_xml(cls, article_json, coding='ascii'):
        # Parse source JSON
        d_article_json = None

        try:
            d_article_json = json.loads(article_json)
        except ValueError as ve:
            errmsg = "Invalid JSON detected: \"{}\". What was obtained: \"{}\".".format(str(ve), article_json)
            raise Exception(errmsg)

        rid_tag = "rid"
        if rid_tag not in d_article_json:
            msg = "No \"{}\" tag exists in source article: \"{}\".".format(rid_tag, article_json)
            raise Exception(msg)
        rid = d_article_json[rid_tag]

        article_blob_tag = 'text-gz-64'
        if article_blob_tag not in d_article_json:
            msg = "No \"{}\" tag exists in source article, rid={}: \"{}\".".format(article_blob_tag, rid, article_json)
            raise Exception(msg)

        article_base64 = d_article_json[article_blob_tag]

        # gzip-ed article XML
        b_article_gzip = None
        try:
            b_article_gzip = base64.b64decode(article_base64)
        except Exception as e:
            msg = "Invalid base64 detected for rid={}: \"{}\". Obtained: \"{}\".".format(rid, str(e), article_base64)
            raise Exception(msg)

        # Uncompress article GZIP
        d_article_xml = None
        try:
            d_article_xml = gzip.decompress(b_article_gzip)
        except Exception as e:
            msg = "Can't decompress GZIP, rid={}: \"{}\".".format(rid, str(e))
            raise Exception(msg)

        # Decode bytes to string.
        article_xml = None
        try:
            article_xml = d_article_xml.decode(coding)
        except Exception as e:
            if coding == 'utf-8':
                msg = "Can't decode article XML, rid={}, to \"{}\" encoding: \"{}\".".format(rid, coding, str(e))
                print(msg)
                raise Exception(msg)

            try:
                article_xml = d_article_xml.decode('utf-8')
                print("Article XML was successfully decoded, rid={}, encoding: \"utf-8\".".format(rid))
            except Exception as ue:
                msg = "Can't decode article XML, rid={}, to \"{}\" encoding: \"utf-8\".".format(rid, str(ue))
                print(msg)
                raise Exception(msg)

        # Get PmId value from just obtained XML
        root_node = None
        try:
            root_node = ET.fromstring(article_xml)
        except Exception as pe:
            msg = "Can't parse decompressed XML: \"{}\". RID: {}.".format(str(pe), rid)
            print(msg)
            raise Exception(msg)

        ln_pmids = root_node.findall('MedlineCitation/PMID')
        if len(ln_pmids) == 0:
            msg = "Can't find \"PubmedArticle/MedlineCitation/PMID\" node in XML. RID: {}.".format(rid)
            print(msg)
            raise Exception(msg)

        pmid = None
        try:
            pmid = int(ln_pmids[0].findtext(".").strip())
        except Exception as ie:
            msg = "Can't convert PmId value to \"int\": \"{}\". PmId: {}, RID: {}.".format(str(ie), pmid, rid)
            print(msg)
            raise Exception(msg)

        return cls(rid=rid, pmid=pmid, xml=article_xml)
# -----------------------------------------------------------------------------------------------------------------------
