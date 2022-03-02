import base64
import zlib
import xml.etree.ElementTree as ET

# -----------------------------------------------------------------------------------------------------------------------


class CInputData(object):
    def __init__(self, pmid: int = None, xml: str = None, json: bytes = None):
        self._pmid = pmid if pmid is not None else None
        self._xml = xml if xml is not None else None

    def pmid(self, pmid: int = None):
        if pmid is not None:
            self._pmid = pmid
        return self._pmid

    def xml(self, xml: str = None):
        if xml is not None:
            self._xml = xml
        return self._xml

    def __str__(self):
        return "|".join([self._pmid, self._xml])

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_json(cls, d_article: dict, coding='ascii'):

        uid_tag = "uid"
        if uid_tag not in d_article:
            msg = "No \"{}\" tag exists in source article: \"{}\".".format(uid_tag, d_article)
            raise Exception(msg)
        uid = d_article[uid_tag]

        article_blob_tag = 'data'
        if article_blob_tag not in d_article:
            msg = "No \"{}\" tag exists in source article, uid={}: \"{}\".".format(article_blob_tag, uid, d_article)
            raise Exception(msg)

        article_base64 = d_article[article_blob_tag]

        # gzip-ed article XML
        b_article_z = None
        try:
            b_article_z = base64.b64decode(article_base64)
        except Exception as e:
            msg = "Invalid base64 detected for rid={}: \"{}\". Obtained: \"{}\".".format(uid, str(e), article_base64)
            raise Exception(msg)

        # Uncompress article GZIP
        d_article_xml = None
        try:
            d_article_xml = zlib.decompress(b_article_z)
        except Exception as e:
            msg = "Can't decompress ZLIB, uid={}: \"{}\".".format(uid, str(e))
            raise Exception(msg)

        # Decode bytes to string.
        article_xml = None
        try:
            article_xml = d_article_xml.decode(coding)
        except Exception as e:
            if coding == 'utf-8':
                msg = "Can't decode article XML, uid={}, to \"{}\" encoding: \"{}\".".format(uid, coding, str(e))
                print(msg)
                raise Exception(msg)

            try:
                article_xml = d_article_xml.decode('utf-8')
                print("Article XML was successfully decoded, uid={}, encoding: \"utf-8\".".format(uid))
            except Exception as ue:
                msg = "Can't decode article XML, uid={}, to \"{}\" encoding: \"utf-8\".".format(uid, str(ue))
                print(msg)
                raise Exception(msg)

        # Get PmId value from just obtained XML
        root_node = None
        try:
            root_node = ET.fromstring(article_xml)
        except Exception as pe:
            msg = "Can't parse decompressed XML: \"{}\". RID: {}.".format(str(pe), uid)
            print(msg)
            raise Exception(msg)

        ln_pmids = root_node.findall('PMID')
        if len(ln_pmids) == 0:
            msg = "Can't find \"MedlineCitation/PMID\" node in XML. UID: {}.".format(uid)
            print(msg)
            raise Exception(msg)

        pmid = None
        try:
            pmid = int(ln_pmids[0].findtext(".").strip())
        except Exception as ie:
            msg = "Can't convert PmId value to \"int\": \"{}\". PmId: {}, UID: {}.".format(str(ie), pmid, uid)
            print(msg)
            raise Exception(msg)

        return cls(pmid=uid, xml=article_xml)
# -----------------------------------------------------------------------------------------------------------------------
