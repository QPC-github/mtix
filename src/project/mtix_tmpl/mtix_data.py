import base64
import sys
import zlib
import xml.etree.ElementTree as ET


class MtixData(object):
    def __init__(self, uid: int = None, data: bytes = None):
        self._uid = uid if uid is not None else None
        self._data = data if data is not None else None
    # ------------------------------------------------------

    def uid(self, uid: int = None):
        if uid is not None:
            self._uid = uid
        return self._uid
    # ------------------------------------------------------

    def data(self, data: bytes = None):
        if data is not None:
            self._data = data
        return self._data
    # ------------------------------------------------------

    def __str__(self):
        return "|".join([str(self._uid), self._data.decode('utf-8')])
    # ------------------------------------------------------

    def __repr__(self):
        return self.__str__()
    # ------------------------------------------------------

    @classmethod
    def from_xml(cls, uid: int, xml: str):
        # Modify and wrap XML: compress using ZLIB and make base64 encode

        # Check if valid MTIX XML is provided and extract "MedlineCitation" from there
        root_node = None
        try:
            root_node = ET.fromstring(xml)
        except Exception as pe:
            msg = "Invalid XML detected: \"{}\". UID: {}.".format(str(pe), uid)
            raise Exception(msg)

        # Extract "MedlineCitation" node
        n_medline_cit = root_node.find('MedlineCitation')
        if n_medline_cit is None:
            msg = "Can't find \"MedlineCitation\" node in XML. UID: {}.".format(uid)
            raise Exception(msg)

        # Get PmId from "MedlineCitation"
        n_pmid = n_medline_cit.find('PMID')
        if n_pmid is None:
            msg = "Can't find \"MedlineCitation/PMID\" node in XML. UID: {}.".format(uid)
            raise Exception(msg)

        # Check if UID and PmId from XML are same
        pmid = None
        try:
            pmid = int(n_pmid.findtext(".").strip())
        except Exception as ie:
            msg = "Can't convert PmId from XML value to \"int\": \"{}\". PmId: {}, UID: {}.".format(str(ie), pmid, uid)
            raise Exception(msg)

        if uid != pmid:
            msg = "\"PmId\" value from MTIX XML differs from \"uid\": {} vs {}. Stop.".format(pmid, uid)
            raise Exception(msg)

        # Remove "<AuthorList>" in place
        MtixData.remove_node(n_medline_cit, "Article/AuthorList", uid)

        # Serialize "MedlineCitation"
        b_medline_citation = ET.tostring(n_medline_cit)

        # Compress XML using ZLIB
        compressed_xml = None
        try:
            b_compressed_xml = zlib.compress(b_medline_citation)
        except Exception as e:
            msg = "Can't compress article XML, uid={}: \"{}\".".format(uid, str(e))
            raise Exception(msg)

        # Apply base64 encoder
        b_base64_xml = None
        try:
            b_base64_xml = base64.b64encode(b_compressed_xml)
        except Exception as e:
            msg = "Can't BASE64 encode for rid={}: \"{}\".".format(uid, str(e))
            raise Exception(msg)

        return cls(uid=uid, data=b_base64_xml)
    # ------------------------------------------------------

    def to_json_dict(self):
        return {"uid": self.uid(), "data": self.data().decode('utf-8')}
# ------------------------------------------------------

    @staticmethod
    def remove_node(n_base_node, xpath2rm: str, uid: int):
        n_node2rm = n_base_node.find(xpath2rm)
        if n_node2rm is None:
            msg = "remove_node(): artcile \"{}\" does not have \"{}\" node. Skip.".format(uid, xpath2rm)
            print(msg, file=sys.stderr)

        # Node exists. Go deep
        n_runner_node = n_base_node
        l_subnodes =  xpath2rm.split('/')

        for idx, subnode in enumerate(l_subnodes):
            n_subnode = n_runner_node.find(subnode)
            if n_subnode is None:
                msg = "remove_node(): can't find \"{}\" in \"{}\". Stop.".format(n_runner_node.tag, subnode)
                raise Exception(msg)

            if (idx == len(l_subnodes) - 1) and (n_subnode.tag == l_subnodes[-1]):
                n_runner_node.remove(n_subnode)
                break
            else:
                n_runner_node = n_subnode
# -----------------------------------------------------------------------------------------------------------------------
