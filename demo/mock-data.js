"use strict";

// The first record is a cleared GreggSpeak test capture and its saved recognition
// result. The second record is fictional for the browser-only workflow.
const DEMO_BATCHES = [
  {
    "id": "GS-2026-001",
    "title": "Case 1",
    "date": "May 2026",
    "processed": true,
    "caseNo": "Case No. 7",
    "witness": "Witness",
    "court": "Municipal Trial Court",
    "hearingDate": "",
    "place": "",
    "pages": [
      {
        "number": 1,
        "confidence": 91.46,
        "rows": 11,
        "words": 72,
        "reviewWords": 5,
        "image": "./assets/project-page-001-scanned.png",
        "segmentationImage": "./assets/project-page-001-segmentation.png",
        "text": "In call the case.\nCLERK:\nCriminal case under seven yes People of the Philippines versus defendant\nfor.\nHearts.\nIs in is appearances please defense counsel is?\nWITNESS:\nFor the prosecution you honor not.\nCLERK:\nFor the defendant you honor not.\nCOURT:\nInside complainant present witness.\nPROSECUTOR:\nYes you honor the complainant yes versus.\nIn call the witness clerk yes court.\nCLERK:\nRaise you right hand and swear to tell the door you yes.\nWITNESS:\nYes not.\nIn proceed yes."
      }
    ]
  },
  {
    id: "GS-2026-002",
    title: "Interview Notes — Practice Session",
    date: "September 2026",
    processed: false,
    caseNo: "Case No. 2026-002",
    witness: "Interviewee",
    court: "Municipal Trial Court",
    hearingDate: "2026-09-01",
    place: "Courtroom 1",
    pages: [
      {
        number: 1,
        confidence: 91.6,
        rows: 2,
        words: 19,
        reviewWords: 1,
        text: "QUESTION: Please describe the document.\nANSWER: It is a copy of the written report.\nQUESTION: When was it prepared?\nANSWER: It was prepared the following morning.",
        wordsSample: [
          { label: "describe", confidence: 94.6, alternatives: ["describe", "description", "described"] },
          { label: "written", confidence: 83.2, alternatives: ["written", "writing", "writer"] },
          { label: "page", confidence: 69.4, alternatives: ["page", "paper", "place"] }
        ]
      }
    ]
  }
];
