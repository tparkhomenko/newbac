LESION_CLASS_MAPPING = {
	"nevus": "benign_only",
	"seborrheic_keratosis": "benign_only",
	"lentigo": "benign_only",
	"vasc": "benign_only",
	"melanoma": "bm",
	"bcc": "bm",
	"scc": "bm",
	"akiec": "bm",
}


def lesion_requires_bm(lesion_class: str) -> bool:
	"""Return True if this lesion class requires the BM head, else False.

	Defaults to True (requires BM) if class is unknown.
	"""
	return (LESION_CLASS_MAPPING.get(str(lesion_class), "bm") == "bm")
